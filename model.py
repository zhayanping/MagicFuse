import os
import sys
from tqdm import tqdm
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.lr_scheduler import StepLR
from utils.config import Config
from utils.loss import Fusionloss
from network import FusionModule
from pretrained.AE.ae import Encoder, Decoder
from pretrained.IKR.unet_module import DiffusionUNet as IKRUNet
from pretrained.CKG.unet_module import DiffusionUNet as CKGUNet
from torch.utils.tensorboard import SummaryWriter

def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == "linear":
        betas = torch.linspace(beta_start, beta_end,
                               num_diffusion_timesteps, dtype=torch.float32)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == torch.Size([num_diffusion_timesteps])
    return betas


class AdaptModel:
    def __init__(self, rank, num_class=9, pretrained=False):
        self.rank = rank
        self.num_class = num_class
        self.ae = nn.Module()
        self.ae.encoder = Encoder()
        self.ae.decoder = Decoder()
        ae_ckpt = torch.load('pretrained/AE/ae.pth', map_location='cpu')
        self.ae.encoder.load_state_dict(ae_ckpt['encoder'])
        self.ae.decoder.load_state_dict(ae_ckpt['decoder'])
        for param in self.ae.encoder.parameters():
            param.requires_grad = False
        for param in self.ae.decoder.parameters():
            param.requires_grad = False

        dcfg = Config(cfg_file='./pretrained/IKR/config.yaml')
        self.IKRModule = IKRUNet(dcfg)
        ikr_ckpt = torch.load('./pretrained/IKR/ikr.pth', map_location='cpu')
        self.IKRModule.load_state_dict(ikr_ckpt, strict=True)
        for param in self.IKRModule.parameters():
            param.requires_grad = False

        scfg = Config(cfg_file='./pretrained/CKG/config.yaml')
        self.CKGModule = CKGUNet(scfg)
        ckg_ckpt = torch.load('./pretrained/CKG/ckg.pth', map_location='cpu')
        self.CKGModule.load_state_dict(ckg_ckpt, strict=True)
        for param in self.CKGModule.parameters():
            param.requires_grad = False

        self.FusionModule = FusionModule(num_class=self.num_class)
        if pretrained:
            ckpt = torch.load(pretrained, map_location='cpu')
            self.FusionModule.load_state_dict(ckpt, strict=True)
            print('==> load pretrained model')

        self.betas = get_beta_schedule(beta_schedule=dcfg.diffusion.beta_schedule,
                                        beta_start=dcfg.diffusion.beta_start,
                                        beta_end=dcfg.diffusion.beta_end,
                                        num_diffusion_timesteps=dcfg.diffusion.num_diffusion_timesteps)
        self.betas = self.betas.to(self.rank)
        alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_pre = torch.cat(
            [torch.ones(1).to(self.rank), self.alphas_cumprod], dim=0)

        self.num_timesteps = dcfg.diffusion.num_diffusion_timesteps
        skip = self.num_timesteps // 25
        self.seq = range(0, self.num_timesteps, skip)
        self.seq_next = [-1] + list(self.seq[:-1])

    def train(self, epochs, train_loader, train_sampler, load_pretrained={}):
        step = 0
        start_epoch = 0
        
        optimizer = torch.optim.SGD(
            self.FusionModule.parameters(),  
            lr=1e-3,         
            momentum=0.9,    
            weight_decay=1e-4 
        )
        
        if load_pretrained:
            ckpt = torch.load(os.path.join(
                load_pretrained['path'], f'ckpt_{load_pretrained['epoch']}.pth'), map_location=f'cuda:{self.rank}')
            opsd = torch.load(os.path.join(
                load_pretrained['path'], f'optimizer_{load_pretrained['epoch']}.pth'), map_location=f'cuda:{self.rank}')
            if isinstance(self.FusionModule, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                self.FusionModule.module.load_state_dict(ckpt, strict=True)
            else:
                self.FusionModule.load_state_dict(ckpt, strict=True)
            start_epoch = opsd['epoch']
            step = opsd['step']
            optimizer.load_state_dict(opsd['optimizer'])
            print(f'==> load weights from {os.path.join(load_pretrained['path'], f'ckpt_{load_pretrained['epoch']}')} '
                  f'epoch:{start_epoch}, step:{step}')
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5, last_epoch=start_epoch-1)
        
        if dist.get_rank() == 0:
            writer = SummaryWriter(log_dir=f'tb-runs')
        else:
            writer = None

        FLoss = Fusionloss(self.rank)
        self.FusionModule.train()
        for epoch in range(start_epoch, epochs):
            train_sampler.set_epoch(epoch)
            for i, (raw, label, name) in enumerate(train_loader):
                batch = raw.shape[0]
               
                raw = raw.to(self.rank)
                label = label.to(self.rank)
                assert torch.all((label >= 0) & (label <= self.num_class-1)), f"Label contains invalid values! Min: {name}"
                raw_z = self.ae.encoder(raw)
                raw_z = data_transform(raw_z)
                b,c,h,w = raw_z.shape

                dege_xt = style_xt = fuse_xt = torch.randn_like(raw_z).to(self.rank)
                seg_xt = torch.randn(b, self.num_class, h, w).to(self.rank)

                select = random.choice(self.seq)

                for it, (i, j) in enumerate(zip(reversed(self.seq), reversed(self.seq_next))):
                    t = (torch.ones(batch) * i).to(self.rank)
                    next_t = (torch.ones(batch) * j).to(self.rank)
                    at = self.alphas_cumprod_pre.index_select(0, t.long()+1).view(-1, 1, 1, 1)
                    at_next = self.alphas_cumprod_pre.index_select(0, next_t.long()+1).view(-1, 1, 1, 1)
                    dege_noise, dege_xt, dege_x0 = self.dege_sample_onestep(raw_z.float(), dege_xt.float(), t, next_t)
                    style_noise, style_xt, style_x0 = self.style_sample_onestep(raw_z.float(), style_xt.float(), t, next_t)
                    with torch.set_grad_enabled((t[0] == select).item()):
                        noise, seg_map, seg_xt = self.FusionModule(torch.cat([dege_x0, style_x0, fuse_xt], dim=1), dege_noise, style_noise, t, seg_xt) 
                    x0 = (fuse_xt - noise * (1 - at).sqrt()) / at.sqrt()
                    c2 = (1 - at_next).sqrt()
                    fuse_xt = at_next.sqrt() * x0 + + c2 * noise
                    if t[0] == select:
                        break
                optimizer.zero_grad()
                degez = inverse_data_transform(dege_x0)
                stylez = inverse_data_transform(style_x0)
                fusez = inverse_data_transform(x0)
                vis_img = self.ae.decoder(degez)
                ir_img = self.ae.decoder(stylez)
                fused_img = self.ae.decoder(fusez)
                # Fusion Loss
                loss_fuse, loss_in, loss_grad, loss_cr, loss_cb = FLoss(fused_img, vis_img, ir_img)
                # Seg Loss
                class_weights = torch.tensor([0.4, 0.8, 1, 2, 4, 1.8, 1.8, 1.5, 1.5]).to(self.rank)
                assert len(class_weights) == self.num_class
                loss_seg = nn.CrossEntropyLoss(weight=class_weights)(seg_map, label.long())
                # total loss
                loss_total = loss_fuse+loss_seg
                loss_total.backward()
                optimizer.step()

                if dist.get_rank() == 0 and step % 10 == 0:
                    sys.stdout.write(f"\r[epoch: {epoch+1}/{epochs}] "
                                    f"[step: {step}] [lr: {optimizer.param_groups[0]['lr']}] "
                                    f"[loss_fuse={loss_fuse:.4f} "
                                    f"loss_seg: {loss_seg:.4f} ] " 
                                    f"[{raw.shape[2], raw.shape[3]}]  "
                                    f"[bs: {train_loader.batch_size}] [t:{select}]")  
                    sys.stdout.flush()
                    writer.add_scalar('loss', loss_total.item(), step)
                    writer.add_scalar('loss_fuse', loss_fuse.item(), step)
                    writer.add_scalar('loss_seg', loss_seg.item(), step)
                if dist.get_rank() == 0:
                    if step % 1000 == 0:
                        try:
                            savepath = 'checkpoints'
                            os.makedirs(savepath, exist_ok=True)
                            state_dict = self.FusionModule.module.state_dict()
                            for key, param in state_dict.items():
                                state_dict[key] = param.cpu()
                            torch.save(state_dict, os.path.join(
                                savepath, f'ckpt_{epoch}.pth'))
                            torch.save({
                                'epoch': epoch + 1,
                                'step': step,
                                'optimizer': optimizer.state_dict(),
                            }, os.path.join(savepath, f'optimizer_{epoch}.pth'))
                            print('==> model saved success')
                        except Exception as e:
                            print(f'==> model saving failed (Error: {str(e)})')
                step += 1
            scheduler.step()
        print('==> Finish Training!')

    def ikr_sample_onestep(self, cond, xt, t, next_t):
        at = self.alphas_cumprod_pre.index_select(0, t.long()+1).view(-1, 1, 1, 1)
        at_next = self.alphas_cumprod_pre.index_select(0, next_t.long()+1).view(-1, 1, 1, 1)
        noise = self.IKRModule(torch.cat([cond, xt], dim=1), t)
        x0 = (xt - noise * (1 - at).sqrt()) / at.sqrt()
        c2 = (1 - at_next).sqrt()
        xt_next = at_next.sqrt() * x0 + c2 * noise
        return noise, xt_next, x0
    
    def ckg_sample_onestep(self, cond, xt, t, next_t):
        at = self.alphas_cumprod_pre.index_select(0, t.long()+1).view(-1, 1, 1, 1)
        at_next = self.alphas_cumprod_pre.index_select(0, next_t.long()+1).view(-1, 1, 1, 1)
        noise = self.CKGModule(torch.cat([cond, xt], dim=1), t)
        x0 = (xt - noise * (1 - at).sqrt()) / at.sqrt()
        c2 = (1 - at_next).sqrt()
        xt_next = at_next.sqrt() * x0 + c2 * noise
        return noise, xt_next, x0

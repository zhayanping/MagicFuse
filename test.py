import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.insert(0, '')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=Warning)
import torch
import dataset
import numpy as np
from PIL import Image
from model import AdaptModel
from torchvision import transforms
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
from sklearn.metrics import confusion_matrix
from torch.nn.parallel import DistributedDataParallel as DDP
from model import data_transform, inverse_data_transform

num_classes = 9
class_name = {0:'unlabelled', 1:'car', 2:'person', 3:'bike', 4:'curve', 5:'car_stop', 6:'guardrail', 7:'color_cone', 8:'bump'}
color_map = {
    0: [0, 0, 0],       # unlabelled
    1: [64, 0, 128],    # car
    2: [64, 64, 0],     # person
    3: [0, 128, 192],   # bike
    4: [0, 0, 192],     # curve
    5: [128, 128, 0],   # car_stop
    6: [64, 64, 128],   # guardrail
    7: [192, 128, 128], # color_cone
    8: [192, 64, 0]     # bump
}  

datapath = 'data'
savepath = 'results'
seg_metric = False

def main(rank, world_size):
    setup(rank, world_size)
    seed = 55
    DATASET = dataset.DATASET(datapath, batchsize=6, mode='test')
    print("=> dataset length:{}".format(len(DATASET.test_dataset)))
    test_loader = DATASET.get_loaders()

    model = AdaptModel(rank=rank, num_class=9, pretrained='./pretrained/MKF/model.pth')
    model.ae.encoder = model.ae.encoder.to(rank)
    model.ae.decoder = model.ae.decoder.to(rank)
    model.IKRModule = model.IKRModule.to(rank)
    model.CKGModule = model.CKGModule.to(rank)
    model.FusionModule = model.FusionModule.to(rank)
    model.FusionModule.eval()

    skip = model.num_timesteps // 25
    seq = range(0, model.num_timesteps, skip)
    seq_next = [-1] + list(seq[:-1])
    global_cm = np.zeros((num_classes, num_classes))
    for raw, label, name in tqdm(test_loader, desc="Inferencing"):
        raw = raw.to(rank)
        label = np.array(label)
        raw_z = model.ae.encoder(raw)
        raw_z = data_transform(raw_z)
        batch, c, h, w = raw_z.shape
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        ikr_xt = ckg_xt = fuse_xt = torch.randn_like(raw_z, dtype=torch.float32).to(rank)  
        seg_xt = torch.randn(batch, 9, h, w).to(rank)
        with torch.no_grad():
            for it, (i, j) in tqdm(enumerate(zip(reversed(seq), reversed(seq_next))),
                                    desc=f"Processing {name}",
                                    leave=False,):
                t = (torch.ones(batch) * i).to(rank)
                next_t = (torch.ones(batch) * j).to(rank)
                at = model.alphas_cumprod_pre.index_select(0, t.long()+1).view(-1, 1, 1, 1)
                at_next = model.alphas_cumprod_pre.index_select(0, next_t.long()+1).view(-1, 1, 1, 1)
                ikr_noise, ikr_xt, ikr_x0 = model.ikr_sample_onestep(raw_z.float(), ikr_xt.float(), t, next_t)
                ckg_noise, ckg_xt, ckg_x0 = model.ckg_sample_onestep(raw_z.float(), ckg_xt.float(), t, next_t)
                noise, seg_map, seg_xt = model.FusionModule(torch.cat([ikr_x0, ckg_x0, fuse_xt], dim=1), ikr_noise, ckg_noise, t, seg_xt) 
                x0 = (fuse_xt - noise * (1 - at).sqrt()) / at.sqrt()
                c2 = (1 - at_next).sqrt()
                fuse_xt = at_next.sqrt() * x0 + c2 * noise
        ikr_z = inverse_data_transform(ikr_x0)
        ckg_z = inverse_data_transform(ckg_x0)
        mkf_z = inverse_data_transform(x0)

        ikrimg = model.ae.decoder(ikr_z)
        ckgimg = model.ae.decoder(ckg_z)
        MagImg = model.ae.decoder(mkf_z)
        img_save(ikrimg.cpu(), name, os.path.join(savepath, 'ikr'))
        img_save(ckgimg.cpu(), name, os.path.join(savepath, 'ckg'))
        img_save(MagImg.cpu(), name, os.path.join(savepath, 'MagImg'))
        
        segmap = torch.softmax(seg_map, dim=1)
        segmap = torch.argmax(segmap, dim=1).cpu() 
        segcolor = visualize_segmentation(segmap) 
        img_save(segcolor.permute(0, 3, 1, 2), name, os.path.join(savepath, 'seg'), scale=False)
        if seg_metric:
            for i in range(b):
                single_label = label[i].flatten()
                single_seglabel = segmap[i].flatten() 
                cm_single = confusion_matrix(single_label, single_seglabel, labels=range(num_classes))
                global_cm += cm_single
    if seg_metric:
        ious = []
        for i in range(num_classes):
            intersection = global_cm[i, i]
            union = global_cm[i, :].sum() + global_cm[:, i].sum() - intersection
            iou = intersection / (union + 1e-8)
            ious.append(iou)
        miou = np.mean(ious)
        print('------------------------------')
        for key in class_name:
            print(f'{class_name[key]:11}\t|\t{ious[key]:.4f}\n')
        print('------------------------------')
        print(f'mIoU:{miou:.4f}\n')
    print('Finished!')

    
def calculate_miou(pred, target, num_classes):
    pred = pred.cpu().numpy() if torch.is_tensor(pred) else np.array(pred)
    target = target.cpu().numpy() if torch.is_tensor(target) else np.array(target)
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    cm = confusion_matrix(target_flat, pred_flat, labels=range(num_classes))
    ious = []
    for i in range(num_classes):
        intersection = cm[i, i]
        union = cm[i, :].sum() + cm[:, i].sum() - intersection
        iou = intersection / (union + 1e-8) 
        ious.append(iou)
    ious = np.array(ious)
    miou = np.nanmean(ious)
    return miou, ious

def img_save(img, name, path, scale=True):
    os.makedirs(path, exist_ok=True)
    if img.dim() == 4: # b c h w
        imgs = torch.unbind(img, dim=0)  
        for img, img_name in zip(imgs, name):
            img = img.permute(1, 2, 0)  # [H, W, C]
            img = img.numpy()
            if scale:
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            if img.shape[2] == 3:
                img = Image.fromarray(img, mode='RGB')
            elif img.shape[2] == 1:
                img = Image.fromarray(img[:, :, 0], mode='L')
            img.save(os.path.join(path, img_name))
    elif img.dim() == 3: # c h w
        img = img.permute(1, 2, 0)  # [H, W, C]
        img = img.numpy()
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        if img.shape[2] == 3:
            img = Image.fromarray(img, mode='RGB')
        elif img.shape[2] == 1:
            img = Image.fromarray(img[:, :, 0], mode='L')
        img.save(os.path.join(path, img_name))
    else:
        raise ValueError('img dim must be 3 or 4')

def visualize_segmentation(label_map):
    b, h, w = label_map.shape
    colored = np.zeros((b, h, w, 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        colored[label_map == class_id] = color
    colored = torch.from_numpy(colored)
    return colored

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356' 
    torch.distributed.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    for i in range(world_size):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    mp.spawn(main, args=(world_size, ), nprocs=world_size, join=True)
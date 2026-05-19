import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)
    

class ResnetBlock(nn.Module):
    def __init__(self, channels, temb_channels, dropout=0.0):
        super().__init__()
        self.channels = channels
        self.norm1 = Normalize(channels)
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels, channels)
        self.norm2 = Normalize(channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return x+h



class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw  (first hw of q, second of k)  w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)
        
        # attend to values
        v = v.reshape(b, c, h*w) # b,c,hw
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        amap = h_.reshape(b, c, h, w)
        h_ = self.proj_out(amap)
        return h_+x


class FusionModule(nn.Module):
    def __init__(self, in_ch=8, channels=96, num_class=9):
        super().__init__()
        self.ch = channels
        self.num_class = num_class
        time_embed_dim = channels*4
        self.time_embed = nn.Sequential(
            nn.Linear(channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.inconv = nn.Sequential(
            nn.Conv2d(in_ch*3, channels, kernel_size=3, padding=1),
            nn.SiLU()
        )
        self.normout = nn.Conv2d(channels, in_ch, kernel_size=1)
        self.estimate = nn.Sequential(
            nn.Conv2d(in_ch*3, channels, kernel_size=3, padding=1),
            nn.Conv2d(channels, in_ch, kernel_size=3, padding=1),
        )
        layernum = 5
        self.blocks = nn.ModuleList()
        for _ in range(layernum):
            block = nn.Module()
            block.res1 = ResnetBlock(channels, time_embed_dim)
            block.res2 = ResnetBlock(channels, time_embed_dim)
            block.mlp = nn.Sequential(
                nn.Conv2d(channels+in_ch*2, channels*2, kernel_size=1),
                nn.SiLU(),
                nn.Conv2d(channels*2, channels, kernel_size=1),
            )
            block.att = AttnBlock(channels)
            self.blocks.append(block)
        self.segproj = nn.ModuleList(
            nn.Conv2d(channels+num_class, num_class, kernel_size=3, padding=1) for _ in range(layernum*2)
        )
        
    def forward(self, x, n1, n2, t, seg):
        h, w = x.shape[2], x.shape[3]
        temb = get_timestep_embedding(t, self.ch)
        temb = self.time_embed(temb)
        x = self.inconv(x)
        att_maps = []
        for block in self.blocks:
            x = block.res1(x, temb)
            x = block.res2(x, temb)
            att_maps.append(x) 
            x = block.mlp(torch.cat([x, n1, n2], dim=1))
            x = block.att(x)
            att_maps.append(x) 
        x =  self.normout(x)
        w_e = self.estimate(torch.cat([x, n1, n2], dim=1)) 

        # segmentation
        for i in range(len(att_maps)):
            seg = self.segproj[i](torch.cat([att_maps[i], seg], dim=1)) 
        
        seg_map = F.interpolate(
            seg, 
            size=(h*4, w*4),      
            mode='bilinear',  
            align_corners=False
        )
        assert seg_map.shape[1]==self.num_class
        # for fuse
        prelabel = torch.softmax(seg, dim=1)
        prelabel = torch.argmax(prelabel, dim=1, keepdim=True)
        mask = (prelabel==2) 
        mask = mask.float()
        w_r = torch.clamp(w_e, max=0.4)*mask+(1-mask)*w_e
        noise = w_r*n1 + (1-w_r)*n2
        return noise, seg_map, seg

   
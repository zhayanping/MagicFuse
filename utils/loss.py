import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_msssim import ssim
from scipy.ndimage import distance_transform_edt

class Fusionloss(nn.Module):
    def __init__(self, rank):
        super(Fusionloss, self).__init__()
        self.sobelconv = Sobelxy().to(rank)

    def forward(self, fusion_image, image_vi, image_ir):
        h, w = image_vi.shape[2], image_vi.shape[3]
        fusion_image = self.RGB2YCrCb(fusion_image)
        image_vi = self.RGB2YCrCb(image_vi)
        image_ir = self.RGB2YCrCb(image_ir)[:, :1]
        # YCrCb
        imagevi_y = image_vi[:, :1]  # [b 1 h w]
        imagevi_cr = image_vi[:, 1:2]
        imagevi_cb = image_vi[:, 2:3]
        fusion_y = fusion_image[:, :1]
        fusion_cr = fusion_image[:, 1:2]
        fusion_cb = fusion_image[:, 2:3]
        assert imagevi_y.shape == image_ir.shape == imagevi_cr.shape
        # loss_in
        in_max = torch.max(imagevi_y, image_ir) 
        loss_in = 6*F.l1_loss(in_max, fusion_y)
        # loss_grad
        viy_grad = self.sobelconv(imagevi_y)
        ir_grad = self.sobelconv(image_ir)
        fusion_grad = self.sobelconv(fusion_y)
        grad_max = torch.max(viy_grad, ir_grad) 
        loss_grad = 10*F.l1_loss(grad_max, fusion_grad)
        # loss_cr
        loss_cr = 5*F.l1_loss(imagevi_cr, fusion_cr)
        # loss_cb
        loss_cb = 5*F.l1_loss(imagevi_cb, fusion_cb)
        # loss_total
        loss = loss_in + loss_grad + loss_cr + loss_cb
        return loss, loss_in, loss_grad, loss_cr, loss_cb

    def RGB2YCrCb(self, input_im):
        im_flat = input_im.transpose(1, 3).transpose(
            1, 2).reshape(-1, 3)  # (nhw,c)
        R = im_flat[:, 0]
        G = im_flat[:, 1]
        B = im_flat[:, 2]
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        Cr = (R - Y) * 0.713 + 0.5
        Cb = (B - Y) * 0.564 + 0.5
        Y = torch.unsqueeze(Y, 1)
        Cr = torch.unsqueeze(Cr, 1)
        Cb = torch.unsqueeze(Cb, 1)
        temp = torch.cat((Y, Cr, Cb), dim=1)
        size = list(input_im.size())
        out = (
            temp.reshape(size[0], size[2], size[3], 3)
            .transpose(1, 3)
            .transpose(2, 3)
        )
        return out
    
    def RGB2Gray(self, x):
        weights = torch.tensor([0.299, 0.587, 0.114], device=x.device)
        gray = (x * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)  
        assert gray.shape==x[:,0:1].shape
        return gray
        

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        weightx = nn.Parameter(data=kernelx, requires_grad=False)
        weighty = nn.Parameter(data=kernely, requires_grad=False)
        self.register_buffer('weightx', weightx)  
        self.register_buffer('weighty', weighty)
    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        grad = torch.abs(sobelx)+torch.abs(sobely)
        return grad
import torch
# import cv2
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from torchvision import transforms
import torch.nn as nn


# Calculate Gram Matrix
def gram_matrix(input):
    b, c, w, h = input.size()
    x = input.view(b, c, w*h)
    x_t = x.transpose(1,2)
    G = torch.matmul(x,x_t)
    # G = torch.mm(x, x.t())

    return G.div(c*w*h)

# Convert RGB image to YUV according to https://en.wikipedia.org/wiki/YUV
def rgb2yuv(img:torch.Tensor):
    img = (img+1)/2     # [-1,1] -> [0,1] 

    r = img[...,0,:,:]
    g = img[...,1,:,:]
    b = img[...,2,:,:]

    y = 0.114*b + 0.299*r + 0.587*g
    u = 0.436*b - 0.147*r - 0.289*g
    v = -0.100*b + 0.615*r - 0.515*g
    return torch.stack((y,u,v),-3)


# Compute BGR mean value for a set of data (no use)
# def compute_mean(data_path):
#     if not os.path.exists(data_path):
#         raise FileNotFoundError(f'Folder {data_path} does not exist')

#     img_paths = os.listdir(data_path)
#     total_sum = np.zeros(3)

#     for img_path in tqdm(img_paths):
#         path = os.path.join(data_path, img_path)
#         img = cv2.imread(path)  # B,G,R
#         total_sum += img.mean(axis=(0, 1))

#     channel_mean = total_sum / len(img_paths)
#     mean = np.mean(channel_mean)
#     bgr_mean = mean - channel_mean
#     bgr_mean = torch.tensor(bgr_mean).float()

#     return bgr_mean

# Save checkpoint
def save_checkpoint(model,epoch,name,args):
    save_dir = f"checkpoint/{args.dataset}/{name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir,f"epoch_{epoch}_gan_{args.gan_type}.pth")
    torch.save(model.state_dict(),save_path)

def init_G_weights(net):
    for m in net.modules():
        if isinstance(m,nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data,a=0.2)

def init_D_weights(net):
    for m in net.modules():
        if isinstance(m,nn.Conv2d):
            nn.init.normal_(m.weight.data,std=0.02)

def calculate_brightness(img):
    R = img[:,:,0].mean()
    G = img[:,:,1].mean()
    B = img[:,:,2].mean()

    brightness = 0.299 * R + 0.587*G + 0.114*B
    return brightness

def adjust_brightness(dst,src):
    dst_brightness = calculate_brightness(dst)
    src_brightness = calculate_brightness(src)

    brightness_scale = src_brightness / dst_brightness

    dst *= brightness_scale
    dst = np.clip(dst,0,255)
    dst = dst.astype(np.uint8)

    return dst

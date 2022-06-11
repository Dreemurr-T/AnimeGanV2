import jittor as jt
import cv2
import os
import numpy as np
from tqdm import tqdm
from jittor import nn
import math

jt.flags.use_cuda = 1

# Calculate Gram Matrix
def gram_matrix(input):
    b, c, w, h = input.size()
    x1 = input.view(b,c, w*h)
    x = jt.array(x1)
    x_t = x.transpose(1,2)
    G = jt.matmul(x,x_t)

    return G/(c*w*h)

# Convert BGR image to YUV according to https://en.wikipedia.org/wiki/YUV
def bgr2yuv(img):
    img = (img+1)/2     # [-1,1] -> [0,1] 

    b = img[...,0,:,:]
    g = img[...,1,:,:]
    r = img[...,2,:,:]
    y = 0.114*b + 0.299*r + 0.587*g
    u = 0.436*b - 0.147*r - 0.289*g
    v = -0.100*b + 0.615*r - 0.515*g
    y = jt.array(y)
    u = jt.array(u)
    v = jt.array(v)
    return jt.misc.stack((y,u,v),-3)


# Save checkpoint
def save_checkpoint(model,epoch,name,args):
    save_dir = f"checkpoint/{args.dataset}/{name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir,f"epoch_{epoch}_batchsize_{args.batch_size}.pkl")
    jt.save(model.state_dict(),save_path)

def init_G_weights(net):
    for m in net.modules():
        if isinstance(m,nn.Conv2d):
            tmp = jt.array(m.weight.data)
            nn.init.kaiming_normal_(tmp, a=0.2)
            m.weight.data = tmp.numpy()

def init_D_weights(net):
    for m in net.modules():
        if isinstance(m,nn.Conv2d):
            tmp = jt.array(m.weight.data)
            jt.normal(tmp,std=0.02)
            m.weight.data = tmp.numpy()

def calculate_brightness(img):
    B = img[:,:,0].mean()
    G = img[:,:,1].mean()
    R = img[:,:,2].mean()

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

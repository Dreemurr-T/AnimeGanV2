import torch
import torch.nn.functional as F
import torch.nn as nn
from model import Vgg19
from utils import gram_matrix, bgr2yuv

# adversarial loss for generator (LSGAN)
def adv_loss_g(generated_logit):
    return torch.mean(torch.square(generated_logit-1))

# content loss for generator
def con_loss_g(vgg,real,generated):
    real_feature_map = vgg(real).detach()
    fake_feature_map = vgg(generated)

    content_loss = nn.L1Loss()(fake_feature_map,real_feature_map)
    return content_loss

#  grayscale style loss for generator
def gs_loss_g(vgg,generated,anime_gray):
    generated_feature_map = vgg(generated)
    gray_feature_map = vgg(anime_gray)

    generated_gram = gram_matrix(generated_feature_map)
    gray_gram = gram_matrix(gray_feature_map)

    style_loss = nn.L1Loss()(generated_gram,gray_gram)
    return style_loss

# color loss for generator
def color_loss_g(real,generated):
    real_yuv = bgr2yuv(real)
    generated_yuv = bgr2yuv(generated)
    
    real_y = real_yuv[:,0,:,:]
    generated_y = generated_yuv[:,0,:,:]

    real_u = real_yuv[:,1,:,:]
    generated_u = generated_yuv[:,1,:,:]

    real_v = real_yuv[:,2,:,:]
    generated_v = generated_yuv[:,2,:,:]

    color_loss = nn.L1Loss()(generated_y,real_y) + nn.HuberLoss()(generated_u,real_u) + nn.HuberLoss()(generated_v,real_v)
    return color_loss

# smooth loss for generator (not mentioned in the paper), see tensorflow version of source code
def smooth_loss_g(args,generated):
    zero_h = torch.zeros(args.batch_size,3,255,256).to(args.device)
    zero_w = torch.zeros(args.batch_size,3,256,255).to(args.device)
    dh = generated[:,:,:-1,:]-generated[:,:,1:,:]
    dw = generated[:,:,:,:-1]-generated[:,:,:,1:]
    
    return nn.MSELoss()(dh,zero_h) + nn.MSELoss()(dw,zero_w)

# loss for discriminator
def discriminator_loss(anime_logit, anime_gray_logit, generated_logit, smooth_logit):
    real_loss = 0
    gray_loss = 0
    fake_loss = 0
    real_blur_loss = 0

    real_loss = torch.mean(torch.square(anime_logit-1))
    gray_loss = torch.mean(torch.square(anime_gray_logit))
    fake_loss = torch.mean(torch.square(generated_logit))
    real_blur_loss = torch.mean(torch.square(smooth_logit))

    # for Hayao : 1.2, 1.2, 1.2, 0.8
    # for Paprika : 1.0, 1.0, 1.0, 0.005
    # for Shinkai: 1.7, 1.7, 1.7, 1.0
    loss = 1.2 * real_loss + 1.2 * fake_loss + \
        1.2 * gray_loss + 0.8 * real_blur_loss

    return loss
import jittor as jt
from jittor import init
from jittor import nn
import numpy as np
from utils import gram_matrix, bgr2yuv

jt.flags.use_cuda = 1

# adversarial loss for generator (LSGAN)
def adv_loss_g(generated_logit):
    return jt.mean(jt.sqr(generated_logit-1))

# content loss for generator
def con_loss_g(vgg,real,generated):
    vgg.eval()
    real_feature_map = vgg(real)
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

    # color_loss = nn.L1Loss()(generated_y,real_y) + nn.HuberLoss()(generated_u,real_u) + nn.HuberLoss()(generated_v,real_v)
    color_loss = nn.L1Loss()(generated_y,real_y) + nn.smooth_l1_loss(generated_u,real_u) + nn.smooth_l1_loss(generated_v,real_v)
    return color_loss

# smooth loss for generator (not mentioned in the paper), see tensorflow version of source code
def smooth_loss_g(args,generated):
    zero_h = jt.zeros((args.batch_size,3,255,256))
    zero_w = jt.zeros((args.batch_size,3,256,255))
    dh = generated[:,:,:-1,:]-generated[:,:,1:,:]
    dw = generated[:,:,:,:-1]-generated[:,:,:,1:]
    
    return nn.MSELoss()(dh,zero_h) + nn.MSELoss()(dw,zero_w)

# loss for discriminator
def discriminator_loss(anime_logit, anime_gray_logit, generated_logit, smooth_logit):
    real_loss = jt.mean(jt.sqr(anime_logit-1))
    gray_loss = jt.mean(jt.sqr(anime_gray_logit))
    fake_loss = jt.mean(jt.sqr(generated_logit))
    real_blur_loss = jt.mean(jt.sqr(smooth_logit))

    # for Hayao : 1.2, 1.2, 1.2, 0.8
    # for Paprika : 1.0, 1.0, 1.0, 0.005
    # for Shinkai: 1.7, 1.7, 1.7, 1.0
    loss = real_loss * 1.2 + fake_loss * 1.2 + gray_loss * 1.2 + real_blur_loss * 1.8

    return loss
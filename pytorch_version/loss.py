import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
from utils import gram_matrix, rgb2yuv
import lpips

loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()

# adversarial loss for generator
def adv_loss_g(args, generated_logit):
    adv_loss = 0
    if args.gan_type == 'wgan-gp' or args.gan_type == 'wgan':
        adv_loss = -torch.mean(generated_logit)
    elif args.gan_type == 'GR':
        adv_loss = torch.mean(nn.BCEWithLogitsLoss()(generated_logit,torch.ones_like(generated_logit)))
    elif args.gan_type == 'lsgan':
        adv_loss = torch.mean(torch.square(1-generated_logit))
    # print(adv_loss.item())
    return adv_loss

# content loss for generator
def con_loss_g(real, generated):

    # real_feature_map = vgg(real).detach()
    # fake_feature_map = vgg(generated)

    # content_loss = nn.L1Loss()(fake_feature_map, real_feature_map)
    real = real.detach()

    content_loss = loss_fn_vgg(generated, real).mean()
    # print(content_loss)
    return content_loss

#  grayscale style loss for generator
def gs_loss_g(vgg, generated, anime_gray):
    generated_feature_map = vgg(generated)
    gray_feature_map = vgg(anime_gray)

    generated_gram = gram_matrix(generated_feature_map)
    gray_gram = gram_matrix(gray_feature_map)

    style_loss = nn.L1Loss()(generated_gram, gray_gram)
    # print(style_loss)
    return style_loss

# color loss for generator
# def color_loss_g(real, generated):
#     real_yuv = rgb2yuv(real)
#     generated_yuv = rgb2yuv(generated)

#     real_y = real_yuv[:, 0, :, :]
#     generated_y = generated_yuv[:, 0, :, :]

#     real_u = real_yuv[:, 1, :, :]
#     generated_u = generated_yuv[:, 1, :, :]

#     real_v = real_yuv[:, 2, :, :]
#     generated_v = generated_yuv[:, 2, :, :]

#     color_loss = nn.L1Loss()(generated_y, real_y) + nn.HuberLoss()(generated_u,
#                                                                    real_u) + nn.HuberLoss()(generated_v, real_v)
#     return color_loss

# smooth loss for generator (not mentioned in the paper), see tensorflow version of source code
def smooth_loss_g(args, generated):
    zero_h = torch.zeros(args.batch_size, 3, 255, 256).to(args.device)
    zero_w = torch.zeros(args.batch_size, 3, 256, 255).to(args.device)
    dh = generated[:, :, :-1, :]-generated[:, :, 1:, :]
    dw = generated[:, :, :, :-1]-generated[:, :, :, 1:]

    return nn.MSELoss()(dh, zero_h) + nn.MSELoss()(dw, zero_w)

# loss for discriminator
def discriminator_loss(args, anime_logit, anime_gray_logit, generated_logit, smooth_logit):
    real_loss = 0
    gray_loss = 0
    fake_loss = 0
    real_blur_loss = 0

    if args.gan_type == 'wgan-gp' or args.gan_type == 'wgan':
        real_loss = -torch.mean(anime_logit)
        gray_loss = torch.mean(anime_gray_logit)
        fake_loss = torch.mean(generated_logit)
        real_blur_loss = torch.mean(smooth_logit)

    elif args.gan_type == 'GR':
        loss = nn.BCEWithLogitsLoss()
        real_loss = torch.mean(loss(anime_logit, torch.ones_like(anime_logit)))
        gray_loss = torch.mean(loss(anime_gray_logit, torch.zeros_like(anime_gray_logit)))
        fake_loss = torch.mean(loss(generated_logit, torch.zeros_like(generated_logit)))
        real_blur_loss = torch.mean(loss(smooth_logit, torch.zeros_like(smooth_logit)))
    
    elif args.gan_type == 'lsgan':
        real_loss = torch.mean(torch.square(anime_logit-1))
        gray_loss = torch.mean(torch.square(anime_gray_logit))
        fake_loss = torch.mean(torch.square(generated_logit))
        real_blur_loss = torch.mean(torch.square(smooth_logit))

    # for Hayao : 1.2, 1.2, 1.2, 0.8
    # for Paprika : 1.0, 1.0, 1.0, 0.005
    # for Shinkai: 1.7, 1.7, 1.7, 1.0
    loss = real_loss + fake_loss + gray_loss + 0.8 * real_blur_loss
    return loss

# def discriminator_loss(args, anime_logit, generated_logit):
#     real_loss = 0
#     fake_loss = 0

#     if args.gan_type == 'wgan-gp' or args.gan_type == 'wgan':
#         real_loss = -torch.mean(anime_logit)
#         fake_loss = torch.mean(generated_logit)

#     elif args.gan_type == 'GR':
#         loss = nn.BCEWithLogitsLoss()
#         real_loss = torch.mean(loss(anime_logit, torch.ones_like(anime_logit)))
#         fake_loss = torch.mean(loss(generated_logit, torch.zeros_like(generated_logit)))

#     loss = real_loss + fake_loss
#     # print(real_loss.item())
#     # print(fake_loss.item())
#     return loss

# wgan-gp improvement(not good, deprecated)
def gradient_penalty(args, D, anime, generated):
    alpha = torch.rand((args.batch_size, 1, 1, 1)).to(args.device)
    alpha = alpha.expand(anime.size())
    interpolated = alpha * anime + (1-alpha) * generated
    interpolated = interpolated.requires_grad_()
    logit = D(interpolated)

    gradients = autograd.grad(outputs=logit,
                              inputs=interpolated,
                              grad_outputs=torch.ones(
                                  logit.size()).to(args.device),
                              create_graph=True,
                              retain_graph=True,
                              )[0]
    gradients = gradients.view(gradients.shape[0], -1)
    gp = ((gradients.norm(2, dim=1) - 1)**2).mean()

    return gp

# Regularized GAN Objective (https://github.com/rothk/Stabilizing_GANs)
def d_regularizer(args,real_logits,real,fake_logits,generated):
    d1 = F.sigmoid(real_logits)
    d2 = F.sigmoid(fake_logits)

    grad_d1 = autograd.grad(outputs=real_logits,
                              inputs=real,
                              grad_outputs=torch.ones(
                                  real_logits.size()).to(args.device),
                              create_graph=True,
                              retain_graph=True,
                              )[0]
    grad_d2 = autograd.grad(outputs=fake_logits,
                              inputs=generated,
                              grad_outputs=torch.ones(
                                  fake_logits.size()).to(args.device),
                              create_graph=True,
                              retain_graph=True,
                              )[0]
    grad_d1 = grad_d1.view(args.batch_size,-1)
    grad_d2 = grad_d2.view(args.batch_size,-1)

    grad_d1_norm = grad_d1.norm(2,dim=1)
    grad_d2_norm = grad_d2.norm(2,dim=1) 

    reg_d1 = torch.square(1-d1) * torch.square(grad_d1_norm)
    reg_d2 = torch.square(d2) * torch.square(grad_d2_norm)

    reg = torch.mean(reg_d1+reg_d2)
    return reg


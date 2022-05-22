import jittor as jt
from jittor import nn
from jittor import init
import argparse
from jittor.dataset import Dataset
from model import Generator, Discriminator, Vgg19
from data import AnimeDataset
from utils import save_checkpoint
from loss import *
import time

jt.flags.use_cuda = 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Hayao')
    parser.add_argument('--data_dir', type=str, default='dataset')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--d_layers', type=int,
                        default=3)    # Number of D layers
    parser.add_argument('--device', type=str, default='cuda')
    # learning rate for generator
    parser.add_argument('--g_lr', type=float, default=8e-5)
    # learning rate for disriminator
    parser.add_argument('--d_lr', type=float, default=16e-5)
    # learning rate for initial training
    parser.add_argument('--init_lr', type=float, default=2e-4)
    parser.add_argument('--init_epoch', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=150)
    # adversial loss weight for generator
    parser.add_argument('--adv_weight_g', type=float, default=200.0)
    # adversial loss weight for discriminator
    parser.add_argument('--adv_weight_d', type=float, default=200.0)
    # content loss weight(1.5 for Hayao, 2.0 for Paprika, 1.2 for Shinkai)
    parser.add_argument('--con_weight', type=float, default=2)
    # style loss weight(2.5 for Hayao, 0.6 for Paprika, 2.0 for Shinkai)
    parser.add_argument('--style_weight', type=float, default=3)
    # color loss weight(15. for Hayao, 50. for Paprika, 10. for Shinkai)
    parser.add_argument('--color_weight', type=float, default=20)
    # smooth loss weight(1. for Hayao, 0.1 for Paprika, 1. for Shinkai)
    parser.add_argument('--smo_weight', type=float, default=1.)
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoint/Haoyao')
    parser.add_argument('--checkpoint_name',type=str,default='epoch_25_batchsize_4.pth')
    parser.add_argument('--if_resume', type=bool, default=False)
    parser.add_argument('--start_epoch',type=int,default=1)

    return parser.parse_args()


def train(args):
    anime_loader = AnimeDataset(args).set_attrs(batch_size=args.batch_size, shuffle=True)
    G = Generator()
    D = Discriminator(args)
    Vgg = Vgg19()

    if args.if_resume:
        G_path = 'checkpoint/Hayao/Generator/epoch_84_batchsize_4.pth'
        D_path = 'checkpoint/Hayao/Discriminator/epoch_84_batchsize_4.pth'
        G.load_state_dict(jt.load(G_path))
        D.load_state_dict(jt.load(D_path))
        print("Model loaded! Continue traning from epoch %d"%(args.start_epoch))

    optimizer_g = nn.Adam(G.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
    optimizer_d = nn.Adam(D.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
    optimizer_init = nn.Adam(
        G.parameters(), lr=args.init_lr, betas=(0.5, 0.999))

    cur_epoch = args.start_epoch + 20
    while cur_epoch <= args.epoch+args.init_epoch:
        if cur_epoch <= args.init_epoch:  # pretrain G
            step = 0
            for train_data in anime_loader:
                start_time = time.time()
                step += 1
                # optimizer_init.zero_grad()
                real_img = train_data[0]
                # real_img = jt.array(real_img)
                # print(real_img.shape)
                # print(real_img)
                # print(real_img.shape)
                generated_img = G(real_img)
                # print(generated_img.shape)
                # print(generated_img)
                init_loss = args.con_weight * con_loss_g(Vgg, real_img, generated_img)
                # print(init_loss[0])
                # init_loss.backward()
                optimizer_init.step(init_loss)
                print("Epoch: %3d Step: %5d / %5d  time: %f s init_loss: %.8f" % (cur_epoch, step, len(anime_loader), time.time() - start_time, init_loss))
        else:  # train AnimeGAN
            step = 0
            """
            G_loss = 0.0
            G_avd_loss = 0.0
            G_con_loss = 0.0
            G_col_loss = 0.0
            G_sty_loss = 0.0
            G_smo_loss = 0.0
            D_loss = 0.0
            """
            for train_data in anime_loader:
                start_time = time.time()
                step += 1
                real_img = train_data[0]    # [-1,1]
                anime = train_data[1]   # [-1,1]
                anime_gray = train_data[2]  # [-1,1]
                anime_smooth = train_data[3]    # [-1,1]

                anime_logit = D(anime)
                anime_gray_logit = D(anime_gray)
                smooth_logit = D(anime_smooth)

                # Train G
                generated_img = G(real_img)
                generated_copy = generated_img.clone().detach()
                generated_logit = D(generated_img)
                # optimizer_g.zero_grad()
                advloss_g = adv_loss_g(generated_logit)
                contentloss_g = con_loss_g(Vgg, real_img, generated_img)
                styleloss_g = gs_loss_g(Vgg, generated_img, anime_gray)
                colorloss_g = color_loss_g(real_img, generated_img)
                smoothloss_g = smooth_loss_g(args, generated_img)

                total_loss_g = args.adv_weight_g*advloss_g + args.con_weight*contentloss_g + \
                    args.style_weight*styleloss_g + args.color_weight * \
                    colorloss_g + args.smo_weight*smoothloss_g
                #total_loss_g.backward()
                optimizer_g.step(total_loss_g)
                """
                G_avd_loss += advloss_g.item()
                G_con_loss += contentloss_g.item()
                G_sty_loss += styleloss_g.item()
                G_smo_loss += smoothloss_g.item()
                G_col_loss += colorloss_g.item()
                G_loss += total_loss_g.item()
                """

                # Train D
                generated_logit_copy = D(generated_copy)
                # optimizer_d.zero_grad()
                loss_d = args.adv_weight_d*discriminator_loss(
                    anime_logit, anime_gray_logit, generated_logit_copy, smooth_logit)
                #loss_d.backward()
                optimizer_d.step(loss_d)
                # D_loss += loss_d.item()

                print("Epoch: %3d Step: %5d / %5d  time: %f s generator_loss: %.8f discriminator_loss: %.8f" % (cur_epoch, step,
                      len(anime_loader), time.time() - start_time, total_loss_g, loss_d))

            # save loss curve
            """
            G_loss /= len(anime_loader)
            G_avd_loss /= len(anime_loader)
            G_con_loss /= len(anime_loader)
            G_sty_loss /= len(anime_loader)
            G_col_loss /= len(anime_loader)
            G_smo_loss /= len(anime_loader)
            D_loss /= len(anime_loader)
            """

        if cur_epoch % 5 == 0 or cur_epoch == args.epoch + 1:
            save_checkpoint(G, cur_epoch-1, name="Generator", args=args)
            save_checkpoint(D, cur_epoch-1, name="Discriminator", args=args)
            print(f"Checkpoint saved at epoch {cur_epoch-1}")
        cur_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    start_time = time.time()
    train(args)
    end_time = time.time()
    train_hour = (end_time-start_time) / 3600.0
    print("Training time: %f" % (train_hour))

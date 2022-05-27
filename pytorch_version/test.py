import os
import argparse
from model import Generator
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
from utils import adjust_brightness

def parse_args():
    parser = argparse.ArgumentParser(description="AnimeGANV2")

    parser.add_argument('--checkpoint_dir',type=str,default='checkpoint/Paprika/Generator/epoch_20_gan_lsgan.pth')
    parser.add_argument('--photo_dir',type=str,default='dataset/test/face_test')
    parser.add_argument('--save_dir',type=str,default='result/Hayao/face_test')
    parser.add_argument('--device',type=str,default='cuda')
    parser.add_argument('--adjust_brightness',type=bool,default=False)

    return parser.parse_args()

def test(args):
    if args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    G = Generator().to(device)
    G.load_state_dict(torch.load(args.checkpoint_dir))
    G.eval()
    photo_names = os.listdir(args.photo_dir)
    for photo_name in tqdm(photo_names):
        photo_path = os.path.join(args.photo_dir,photo_name)
        photo = Image.open(photo_path)
        photo_copy = transforms.ToTensor()(photo)
        photo_copy = photo_copy.unsqueeze(0).to(device)
        photo_copy = photo_copy * 2 - 1
        generated_photo = G(photo_copy)
        save_photo = generated_photo.squeeze(0).detach().cpu()
        save_photo = (save_photo + 1) / 2
        save_photo = transforms.ToPILImage()(save_photo)
        if args.adjust_brightness:
            save_path = adjust_brightness(save_photo,photo)
        save_path = os.path.join(args.save_dir,photo_name)
        save_photo.save(save_path)
        
if __name__ == '__main__':
    args = parse_args()
    test(args=args)

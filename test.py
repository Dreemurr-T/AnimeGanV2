import os
import argparse
from model import Generator
from tqdm import tqdm
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms

def parse_args():
    parser = argparse.ArgumentParser(description="AnimeGANV2")

    parser.add_argument('--checkpoint_dir',type=str,default='checkpoint/weights/celeba_distill.pt')
    parser.add_argument('--photo_dir',type=str,default='dataset/test/HR_photo')
    parser.add_argument('--save_dir',type=str,default='result/Hayao/HR_photo')

    return parser.parse_args()

def test(args):
    if torch.cuda.is_available():
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
        photo = cv2.imread(photo_path)
        photo = transforms.ToTensor()(photo)
        photo = photo.unsqueeze(0).to(device)
        generated_photo = G(photo)
        save_photo = generated_photo.squeeze(0).detach().cpu().numpy().transpose(1,2,0)
        save_photo = (save_photo+1.0)/2.0*255
        print(save_photo)
        save_path = os.path.join(args.save_dir,photo_name)
        cv2.imwrite(save_path,save_photo)
        
if __name__ == '__main__':
    args = parse_args()
    test(args=args)

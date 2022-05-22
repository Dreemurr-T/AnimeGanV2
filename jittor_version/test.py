import jittor as jt
from jittor import init
from jittor import nn
from jittor import transform
import os
import argparse
from model import Generator
from tqdm import tqdm
import numpy as np
import cv2
from utils import adjust_brightness

jt.flags.use_cuda = 1

def parse_args():
    parser = argparse.ArgumentParser(description="AnimeGANV2")

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint/Hayao/Generator/epoch_159_batchsize_4.pth')
    parser.add_argument('--photo_dir', type=str, default='dataset/samples/inputs/')
    parser.add_argument('--save_dir', type=str, default='result/Hayao/samples')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--adjust_brightness', type=bool, default=True)

    return parser.parse_args()


def test(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    G = Generator()
    G.load_state_dict(jt.load(args.checkpoint_dir))
    G.eval()
    photo_names = os.listdir(args.photo_dir)
    for photo_name in tqdm(photo_names):
        photo_path = os.path.join(args.photo_dir, photo_name)
        photo = cv2.imread(photo_path)
        photo_copy = transform.to_tensor(photo)
        photo_copy = photo_copy.unsqueeze(0)
        photo_copy = photo_copy * 2 - 1
        generated_photo = G(photo_copy)
        save_photo = generated_photo.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
        save_photo = (save_photo + 1.0) / 2.0 * 255
        save_photo = np.clip(save_photo, 0, 255)
        if args.adjust_brightness:
            save_path = adjust_brightness(save_photo, photo)
        save_path = os.path.join(args.save_dir, photo_name)
        cv2.imwrite(save_path, save_photo)


if __name__ == '__main__':
    args = parse_args()
    test(args=args)


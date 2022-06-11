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
import time

jt.flags.use_cuda = 1

def parse_args():
    parser = argparse.ArgumentParser(description="AnimeGANV2")

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint/Hayao/Generator/epoch_59_batchsize_2.pkl')
    parser.add_argument('--photo_dir', type=str, default='dataset/samples/inputs/')
    parser.add_argument('--save_dir', type=str, default='result/Hayao/samples')
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
        photo_copy = np.transpose(photo_copy,(2,0,1))
        photo_copy = jt.array(photo_copy)
        photo_copy = jt.unsqueeze(photo_copy,0)
        photo_copy = photo_copy * 2 - 1
        generated_photo = G(photo_copy)
        save_photo = jt.squeeze(generated_photo,0)
        save_photo = save_photo.numpy()
        save_photo = np.transpose(save_photo,(1, 2, 0))
        save_photo = (save_photo + 1.0) / 2.0 * 255
        save_photo = np.clip(save_photo, 0, 255)
        if args.adjust_brightness:
            save_path = adjust_brightness(save_photo, photo)
        save_path = os.path.join(args.save_dir, photo_name)
        cv2.imwrite(save_path, save_photo)


if __name__ == '__main__':
    args = parse_args()
    start_time = time.time()
    test(args=args)
    end_time = time.time()
    test_time = (end_time - start_time) / 60.0
    print("Testing time: %f" % (test_time))


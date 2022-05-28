import os
# import cv2
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
# from utils import compute_mean
from torchvision import transforms

"""
folder structure:
    - {data_dir}
        - photo
            1.jpg, ..., n.jpg
        - {dataset}  # E.g Hayao
            smooth
                1.jpg, ..., n.jpg
            style
                1.jpg, ..., n.jpg
"""


class AnimeDataset(Dataset):
    def __init__(self, args):
        if args.device == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.data_dir = args.data_dir
        dataset = args.dataset

        anime_dir = os.path.join(args.data_dir, dataset)
        if not os.path.exists(anime_dir):
            raise FileNotFoundError(f'Folder {anime_dir} does not exist')

        # self.bgr_mean = compute_mean(os.path.join(anime_dir, 'style')) / 255

        self.img = {}
        self.train_photo = 'debug'
        self.style = f'{dataset}/style'
        self.smooth = f'{dataset}/smooth'

        for img_type in [self.train_photo, self.style, self.smooth]:
            img_folder = os.path.join(self.data_dir, img_type)
            print(img_folder)
            img_names = os.listdir(img_folder)

            self.img[img_type] = [os.path.join(
                img_folder, img_name) for img_name in img_names]

        print(
            f'Dataset: real {len(self.img[self.train_photo])} style {self.anime_len}, smooth {self.smooth_len}')

    def __len__(self):
        return len(self.img[self.train_photo])

    @property
    def anime_len(self):
        return len(self.img[self.style])

    @property
    def smooth_len(self):
        return len(self.img[self.smooth])

    def __getitem__(self, index):
        image = self.load_img(index)
        anime_index = index
        if anime_index > self.anime_len-1:
            anime_index -= self.anime_len * (index//self.anime_len)

        anime, anime_gray = self.load_anime(anime_index)
        # anime = self.load_anime(anime_index)
        smooth_gray = self.load_smooth(anime_index)

        return image.to(self.device), anime.to(self.device), anime_gray.to(self.device), smooth_gray.to(self.device)
        # return image.to(self.device), anime.to(self.device), anime_gray.to(self.device)

    def load_img(self, index):
        file_path = self.img[self.train_photo][index]
        image = Image.open(file_path)
        image = self._transform(image)
        return image

    def load_anime(self, index):
        file_path = self.img[self.style][index]
        image = Image.open(file_path)
        image_gray = image.copy().convert('L')
        image_gray = np.stack([image_gray, image_gray, image_gray], axis=-1)
        image_gray = self._transform(image_gray)

        image = self._transform(image)

        # return image, image_gray
        return image, image_gray

    def load_smooth(self, index):
        file_path = self.img[self.smooth][index]
        image = Image.open(file_path).convert('L')
        image = np.stack([image, image, image], axis=-1)
        image = self._transform(image)
        return image

    def _transform(self, img, addMean=False):
        img = transforms.ToTensor()(img)    # [0,255] -> [0,1]
        img = (img*2.0)-1 # [0,1] -> [-1,1]
        # if addMean:
        #     for i in range(img.shape[0]):
        #         img[i] += self.bgr_mean[i]
        return img

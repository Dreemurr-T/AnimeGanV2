import jittor as jt
import jittor.dataset
from jittor import init
from jittor import nn
import os
import cv2
import numpy as np
from jittor.dataset import Dataset
from jittor import transform

jt.flags.use_cuda = 1

class AnimeDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.batch_size = args.batch_size
        self.shuffle = True
        self.data_dir = args.data_dir
        dataset = args.dataset
        anime_dir = os.path.join(args.data_dir, dataset)
        if (not os.path.exists(anime_dir)):
            raise FileNotFoundError(f'Folder {anime_dir} does not exist')
        self.img = {}
        self.train_photo = 'train_photo'
        self.style = f'{dataset}/style'
        self.smooth = f'{dataset}/smooth'
        for img_type in [self.train_photo, self.style, self.smooth]:
            img_folder = os.path.join(self.data_dir, img_type)
            print(img_folder)
            img_names = os.listdir(img_folder)
            self.img[img_type] = [os.path.join(img_folder, img_name) for img_name in img_names]
        print(f'Dataset: real {len(self.img[self.train_photo])} style {self.anime_len}, smooth {self.smooth_len}')
        self.set_attrs(batch_size=self.batch_size, shuffle=self.shuffle)

    def __len__(self):
        return len(self.img[self.train_photo])

    @property
    def anime_len(self):
        return len(self.img[self.style])

    @property
    def smooth_len(self):
        return len(self.img[self.smooth])

    def __getitem__(self, index):
        # print("hello")
        image = self.load_img(index)
        anime_index = index
        if (anime_index > (self.anime_len - 1)):
            anime_index -= (self.anime_len * (index // self.anime_len))
        (anime, anime_gray) = self.load_anime(anime_index)
        smooth_gray = self.load_smooth(anime_index)
        return (image, anime, anime_gray, smooth_gray)

    def load_img(self, index):
        file_path = self.img[self.train_photo][index]
        image = cv2.imread(file_path)
        image = self._transform(image)
        return image

    def load_anime(self, index):
        file_path = self.img[self.style][index]
        image = cv2.imread(file_path)
        image_gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        image_gray = np.stack([image_gray, image_gray, image_gray], axis=(- 1))
        image_gray = self._transform(image_gray)
        image = self._transform(image)
        return (image, image_gray)

    def load_smooth(self, index):
        file_path = self.img[self.smooth][index]
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        image = np.stack([image, image, image], axis=(- 1))
        image = self._transform(image)
        return image

    def _transform(self, img):
        img = transform.to_tensor(img)
        img = np.transpose(img,(2,0,1))
        # img = np.float32(img) * np.float32(1/255.0)
        img = ((img * 2.0) - 1)
        return jt.array(img)

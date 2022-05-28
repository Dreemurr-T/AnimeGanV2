import os
import shutil

from_dir = 'dataset/train_photo'
to_dir = 'dataset/train'

if not os.path.exists(to_dir):
    os.makedirs(to_dir)

name_list = os.listdir(from_dir)[:2400]

for name in name_list:
    shutil.copy(os.path.join(from_dir, name), os.path.join(to_dir, name))
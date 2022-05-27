import os
import shutil

from_dir = 'dataset/celebA_256'
to_dir = 'dataset/test/face_test'

if not os.path.exists(to_dir):
    os.makedirs(to_dir)

name_list = os.listdir(from_dir)[4800:6000]

for name in name_list:
    shutil.copy(os.path.join(from_dir, name), os.path.join(to_dir, name[:-4]))
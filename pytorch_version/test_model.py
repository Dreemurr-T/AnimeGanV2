from matplotlib import transforms
from model import Vgg19
import cv2
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from utils import gram_matrix,bgr2yuv

vgg_mean = torch.tensor([0.485, 0.456, 0.406]).float()
vgg_std = torch.tensor([0.229, 0.224, 0.225]).float()

transformer = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize((224, 224)),
    # transforms.Normalize(vgg_mean, vgg_std)
])

vgg = Vgg19().eval().cuda()
img = cv2.imread('dataset/Hayao/style/0.jpg')
img2 = cv2.imread('dataset/Hayao/style/2.jpg')

print(img.transpose(2,0,1)/255)

# img = img / 127.5 - 1.0
img = transformer(img).unsqueeze(0).cuda()

print(img)
# # img_yuv = bgr2yuv(img)
# feature_map = vgg(img)
# print(feature_map)
# gram = gram_matrix(feature_map)
# print(gram)

# img2 = transformer(img2).unsqueeze(0).cuda()
# test_img = torch.cat((img,img2),0)
# gram = gram_matrix(test_img)
# print(gram)
# print(gram.shape)
# feature = vgg(img)
# print(feature.shape)

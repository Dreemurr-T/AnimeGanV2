import cv2 as cv
from PIL import Image
import torchvision.transforms as transforms 

transformer = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

face = 'dataset/face_train/000004.jpg'
face = cv.imread(face)

face2 = 'dataset/face_train/000004.jpg'
face2 = Image.open(face2)

face_t = transformer(face2)
face_t = face_t * 2 - 1
print(face_t.dtype)


print(face.shape)
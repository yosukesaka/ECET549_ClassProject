import albumentations as A
import cv2
import os

# setup the transformation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.7),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(p=0.3),
    A.Blur(blur_limit=3, p=0.3),
], bbox_params=A.BboxParams(format='yolo'))

# augment the image
image = cv2.imread('real_hanger_7.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# augment the image 10 times and save the augmented images
for i in range(20, 30):
    augmented = transform(image=image)
    aug_image = augmented['image']
    cv2.imwrite(f'real_aug_hanger_{i}.png', cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
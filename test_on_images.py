import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import cv2
import glob
import numpy as np
from PIL import Image

from core.utils import load_image, deprocess_image, preprocess_image
from core.networks import unet_spp_swish_generator_model
from core.dcp import estimate_transmission


img_size = 512


def preprocess_image(cv_img):
    cv_img = cv2.resize(cv_img, (img_size,img_size))
    img = np.array(cv_img)
    img = (img - 127.5) / 127.5
    return img


def load_image(path):
    img = Image.open(path)
    return img


def deprocess_image(img):
    img = img * 127.5 + 127.5
    return img.astype('uint8')


def get_file_name(path):
    basename = os.path.basename(path)
    onlyname = os.path.splitext(basename)[0]
    return onlyname


def preprocess_cv2_image(cv_img):
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    cv_img = cv2.resize(cv_img, (img_size, img_size))
    img = np.array(cv_img)
    img = (img - 127.5) / 127.5
    return img


def preprocess_depth_img(cv_img):
    cv_img = cv2.resize(cv_img, (img_size, img_size))
    img = np.array(cv_img)
    img = np.reshape(img, (img_size, img_size, 1))
    img = 2*(img - 0.5)
    return img



g = unet_spp_swish_generator_model()
weight_path = "./weights/g/generator_30_81.h5"
g.load_weights(weight_path)
g.summary()


output_dir = "outputs/generator_30_81"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if __name__ == "__main__":

    img_src = glob.glob("dataset/nhhaze/val_A/*.png")

    cnt=0
    for img_path in img_src:

        img_name = get_file_name(img_path)
        ori_image = cv2.imread(img_path)
        h, w, _ = ori_image.shape

        ori_image = cv2.resize(ori_image, (img_size,img_size))

        t = estimate_transmission(ori_image)
        t = preprocess_depth_img(t)

        ori_image = preprocess_cv2_image(ori_image)
        x_test = np.concatenate((ori_image, t), axis=2)
        x_test = np.reshape(x_test, (1,img_size,img_size,4))

        generated_images = g.predict(x=x_test)[0]
        generated_images = np.array(generated_images)

        de_test = deprocess_image(generated_images)
        de_test = np.reshape(de_test, (img_size,img_size,3))
        de_test = cv2.resize(de_test, (w, h))

        rgb_de_test = cv2.cvtColor(de_test, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{output_dir}/{img_name}.jpg", rgb_de_test)

    print("Done!")


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import cv2
import glob
import numpy as np
import math
import timeit

from tqdm import tqdm
from PIL import Image

from core.utils import load_image, deprocess_image, preprocess_image
from core.networks import unet_spp_swish_generator_model
from core.dcp import estimate_transmission
from skimage.metrics import structural_similarity as cal_ssim


img_size = 512
RESHAPE = (img_size,img_size)


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


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def preprocess_cv2_image(cv_img):
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    cv_img = cv2.resize(cv_img, RESHAPE)
    img = np.array(cv_img)
    img = (img - 127.5) / 127.5
    return img


def preprocess_depth_img(cv_img):
    cv_img = cv2.resize(cv_img, RESHAPE)
    img = np.array(cv_img)
    img = np.reshape(img, (RESHAPE[0], RESHAPE[1], 1))
    img = 2*(img - 0.5)
    return img



if __name__ == "__main__":

    img_src = glob.glob("dataset/nhhaze/val_A/*.png")

    weight_src = glob.glob("./weights/g/*.h5")


    test_imgs = []
    label_imgs = []

    data_cnt=0
    for img_path in img_src:

        img_name = get_file_name(img_path)

        sharp_img = cv2.imread("dataset/nhhaze/val_B/" + img_name + ".png")
        sharp_img = cv2.cvtColor(sharp_img, cv2.COLOR_BGR2RGB)
        sharp_img = cv2.resize(sharp_img, (img_size,img_size))

        label_imgs.append(sharp_img)

        ori_image = cv2.imread(img_path)
        h, w, _ = ori_image.shape

        ori_image = cv2.resize(ori_image, RESHAPE)

        t = estimate_transmission(ori_image)
        t = preprocess_depth_img(t)

        ori_image = preprocess_cv2_image(ori_image)
        x_test = np.concatenate((ori_image, t), axis=2)
        x_test = np.reshape(x_test, (1,img_size,img_size,4))
        test_imgs.append(x_test)
        data_cnt+=1
        print("Loaded " + str(data_cnt) + "/" + str(len(img_src)))

    w_th = 0

    for weight_path in tqdm(weight_src):

        txtfile = open("model_test_log.txt", "a+")

        model_name = get_file_name(weight_path)
        w_th+=1

        g = unet_spp_swish_generator_model()
        g.load_weights(weight_path)

        psnrs = []
        ssims = []
        totaltime=0

        cnt=0

        for i in range(len(test_imgs)):

            x_test = test_imgs[i]
            sharp_img = label_imgs[i]

            start = timeit.default_timer()
            generated_images = g.predict(x=x_test)[0]
            end = timeit.default_timer()
            infertime = end-start
            if cnt==0: infertime=0
            totaltime+=float(infertime)

            generated_images = np.array(generated_images)

            de_test = deprocess_image(generated_images)
            de_test = np.reshape(de_test, (img_size,img_size,3))

            psnr = cv2.PSNR(de_test, sharp_img)
            ssim = cal_ssim(sharp_img, de_test, data_range=de_test.max() - de_test.min(), multichannel=True)

            psnrs.append(psnr)
            ssims.append(ssim)

            cnt+=1

        average_psnr = np.mean(np.array(psnrs), axis=-1)
        average_ssim = np.mean(np.array(ssims), axis=-1)
        average_time = totaltime/(len(img_src)-1)

        print("Model Name: " + model_name +  " - PSNR: " + str(average_psnr) +  " - SSIM: " + str(average_ssim) + " - Time: " + str(average_time), file=txtfile)

        txtfile.close()
        
    print("Done!")
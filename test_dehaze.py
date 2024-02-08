import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import cv2
import glob
import time
import argparse
import numpy as np

from PIL import Image
from core.utils import deprocess_image, preprocess_cv2_image, preprocess_depth_img, get_file_name
from core.networks import SDDN
from core.dcp import estimate_transmission



if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Testing SDDN Model')
    parser.add_argument('--image_path', type=str, help='path to test image folder', required=True)
    parser.add_argument('--model_path', type=str, help='path to pre-trained model', required=True)
    parser.add_argument('--output_path', type=str, help='path to output image folder', required=True)
    parser.add_argument('--image_size', type=int, help='image size', default=512)
    parser.add_argument('--ch_mul', type=float, help='channel multiplier', default=0.25)
    args = parser.parse_args()


    weight_path = args.model_path
    folder_path = args.image_path
    img_size = args.image_size
    g = SDDN(ch_mul=args.ch_mul)
    g.load_weights(weight_path)
    g.summary()


    output_dir = args.output_path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_src = glob.glob(f'{folder_path}/*')

    for i, img_path in enumerate(img_src):

        img_name = get_file_name(img_path)
        ori_image = cv2.imread(img_path)
        h, w, _ = ori_image.shape

        ori_image = cv2.resize(ori_image, (img_size,img_size))

        start = time.time()
        t = estimate_transmission(ori_image)
        t = preprocess_depth_img(t)

        ori_image = preprocess_cv2_image(ori_image)
        x_test = np.concatenate((ori_image, t), axis=2)
        x_test = np.reshape(x_test, (1,img_size,img_size,4))

        generated_images = g.predict(x=x_test)[0]
        end = time.time()
        proc_time = end - start
        print(f'Img {i+1}: Time = {proc_time}')

        generated_images = np.array(generated_images)

        de_test = deprocess_image(generated_images)
        de_test = np.reshape(de_test, (img_size,img_size,3))
        de_test = cv2.resize(de_test, (w, h))

        rgb_de_test = cv2.cvtColor(de_test, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{output_dir}/{img_name}.jpg", rgb_de_test)

    print("Done!")


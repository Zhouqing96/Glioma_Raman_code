from skimage.metrics import structural_similarity as SSIM
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt


def read_img(path):
    return Image.open(path)


def kmeans(path):
    img = cv2.imread(path)
    data = img.reshape((-1, 3))
    data = np.float32(data)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers4 = cv2.kmeans(data, 4, None, criteria, 10, flags)
    centers4 = np.uint8(centers4)
    res = centers4[labels.flatten()]
    center = centers4[3]
    for j in range(len(res)):
        if (res[j] == center).all():
            continue
        else:
            res[j] = [255, 255, 255]
    dst4 = res.reshape(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dst = cv2.cvtColor(dst4, cv2.COLOR_BGR2RGB)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    cv2.imshow('dst4', dst4)
    cv2.waitKey(0)
    save_path = "./new/Raman/3.jpg"
    cv2.imwrite(save_path, dst4)


if __name__ == "__main__":
    # If the input is a multichannel (color) image, set multichannel=True.
    img1_path = "./SSIM/IF黑.jpg"
    img2_path = "./SSIM/Raman黑.jpg"
    img1 = read_img(img1_path)
    img2 = read_img(img2_path)
    arr1 = np.array(img1)
    arr2 = np.array(img2.resize(img1.size))
    cv2.imshow('111', arr1)
    cv2.imshow('222', arr2)
    cv2.waitKey(0)
    print("The SSIM is:\n", SSIM(arr1, arr2, multichannel=True))

import cv2
import numpy as np

def randSelect(img, nw, nh):
    w = img.shape[0]
    h = img.shape[1]

    M = cv2.getRotationMatrix2D((w / 2, h / 2), np.random.uniform(-20, 20), 1)
    img = cv2.warpAffine(img, M, (w, h))
    w,h = img.shape[0], img.shape[1]

    iw = np.random.randint(0, w - nw)
    ih = np.random.randint(0, h - nh)
    rImg = img[iw:iw+nw, ih: ih + nh, :]
    rImg = cv2.cvtColor(rImg, cv2.COLOR_RGB2GRAY)
    rImg = rImg.astype(np.float32)
    rImg = (rImg - 128)/100
    return rImg

def select2Test(img, nw, nh):
    w = img.shape[0]
    h = img.shape[1]
    iw =  (w - nw)/2
    ih =  (h - nh)/2
    rImg = img[iw:iw + nw, ih: ih + nh, :]
    rImg = cv2.cvtColor(rImg, cv2.COLOR_RGB2GRAY)
    rImg = rImg.astype(np.float32)
    rImg = (rImg - 128) / 100
    return rImg

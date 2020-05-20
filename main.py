from Averaging import encodeAvg, decodeAvg
from arithmetic import decodeArithmetic, encodeArithmetic
from dct import applyDCT, applyIDCT, removeZerosAfterDCT, addZerosForIDCT
from preProcessing import preProcess
import cv2
import numpy as np

img = cv2.imread('yo.png', 0)
rows, cols = img.shape
cv2.imshow('image', img)
print(rows*cols)
#arr = np.array([0])

#for i in range(rows):
 #   for j in range(cols):
  #      arr = np.append(arr, img[i, j])
#arr = np.delete(arr, 0)
#preProcess(arr)
#encodedArr, possibilitiesArr = encodeArithmetic(arr, 4)
#arr = decodeArithmetic(encodedArr, possibilitiesArr, 4, arr.size)

avgArr = encodeAvg(img)
preProcess(avgArr)
np.save("Encoded Image.npy", avgArr)

zerosImage = np.zeros_like(img)
decodeAvg(zerosImage, avgArr)
img = zerosImage

Gaussian = cv2.GaussianBlur(img, (3, 3), 0)
cv2.imshow('Gaussian Blurring', Gaussian)

#k = 0
#for i in range(rows):
 #   for j in range(cols):
  #      img[i, j] = arr[k]
   #     k += 1

arr = applyDCT(img)
arr = removeZerosAfterDCT(arr, rows, cols)
print(arr, arr.size)

zerosImage = np.zeros_like(img)
arr = addZerosForIDCT(arr, img)
img = applyIDCT(arr, zerosImage)
img = cv2.GaussianBlur(img, (3, 3), 0)
cv2.imshow('Imagaya', img)
cv2.waitKey(0)

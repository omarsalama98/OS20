from arithmetic import decodeArithmetic, encodeArithmetic
from dct import applyDCT, applyIDCT, removeZerosAfterDCT, addZerosForIDCT
from preProcessing import preProcess
import cv2
import numpy as np

img = cv2.imread('bika.png', 0)
rows, cols = img.shape
arr = np.array([0])

for i in range(rows):
    for j in range(cols):
        arr = np.append(arr, img[i, j])
arr = np.delete(arr, 0)
preProcess(arr)
encodedArr, possibilitiesArr = encodeArithmetic(arr, 4)
arr = decodeArithmetic(encodedArr, possibilitiesArr, 4, arr.size)

k = 0
for i in range(rows):
    for j in range(cols):
        img[i, j] = arr[k]
        k += 1

arr = applyDCT(img)
arr = removeZerosAfterDCT(arr, rows, cols)
arr = addZerosForIDCT(arr)
zerosImage = np.zeros_like(img)
img = applyIDCT(arr, zerosImage)

cv2.imshow('imag', img)
cv2.waitKey(0)

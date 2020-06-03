import cv2
import numpy as np

from Averaging import encodeAverage, decodeAverage
from Huffman import encodeHuff, decodeHuff
from PreProcessing import preProcess

img = cv2.imread('bika.png', 0)
rows, cols = img.shape
print(rows * cols)
avgArr, lSize, rSize, tSize, bSize = encodeAverage(img, 4, 2, 6)
preProcess(avgArr)
print(avgArr.size)
encodedArr, myDictionary = encodeHuff(avgArr)
np.save("Encoded Image.npy", encodedArr)
np.save("Dictionary.npy", myDictionary)

avgArr = decodeHuff(encodedArr, myDictionary)
imag = decodeAverage(avgArr, lSize, rSize, tSize, bSize, 4, 2, 6, rows, cols)

Gaussian = cv2.GaussianBlur(imag, (3, 3), 0)
cv2.imwrite("Decoded Image.png", imag)

cv2.imshow('Gaussian Blurring', np.hstack([img, imag, Gaussian]))
cv2.waitKey(0)

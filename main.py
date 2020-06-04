import cv2
import numpy as np

from Averaging import encodeAverage, decodeAverage
from DeepLearning import applyDL
from Huffman import encodeHuff, decodeHuff
from PreProcessing import preProcess

bestStartX, bestStartY, bestEndX, bestEndY = applyDL(cv2.imread('doola.jpg'))
image = cv2.imread('doola.jpg', 0)
rows, cols = image.shape

print(rows * cols)
avgArr, lSize, rSize, tSize, bSize = encodeAverage(image, bestStartX, bestEndX, bestStartY, bestEndY, 1)
preProcess(avgArr, 3)
print(avgArr.size)
encodedArr, myDictionary = encodeHuff(avgArr)
np.save("Encoded Image.npy", encodedArr)
np.save("Dictionary.npy", myDictionary)

avgArr = decodeHuff(encodedArr, myDictionary)
imag = decodeAverage(avgArr, lSize, rSize, tSize, bSize, rows, cols, bestStartX, bestEndX, bestStartY, bestEndY, 1)

cv2.imwrite("Decoded Image.png", imag)

cv2.imshow('OS20', np.hstack([imag, image]))
cv2.waitKey(0)

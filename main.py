import cv2
import numpy as np

from Averaging import encodeAverage, decodeAverage
from DeepLearning import applyDL
from Huffman import encodeHuff, decodeHuff
from PreProcessing import preProcess

testImage = 'test4.jpg'

# Using Deep Learning model for setting my bounds.
bestStartX, bestStartY, bestEndX, bestEndY = applyDL(cv2.imread(testImage))

image = cv2.imread(testImage, 0)
rows, cols = image.shape
cv2.imshow('Original', image)
print(rows * cols)

# Averaging the pixels around the bounded object.
avgArr, lSize, rSize, tSize, bSize = encodeAverage(image, bestStartX, bestEndX, bestStartY, bestEndY, 1)

# Quantifying the values into three values each 10 integers.
preProcess(avgArr, 4)
print(avgArr.size)

# Applying Huffman
encodedArr, myDictionary = encodeHuff(avgArr)

numArray = np.array(1)
numArray = np.delete(numArray, 0)
numArray = np.append(numArray, (lSize, rSize, tSize, bSize, rows, cols, bestStartX, bestEndX, bestStartY, bestEndY))
np.save("Encoded Image.npy", encodedArr)
np.save("Dictionary.npy", myDictionary)
np.save("nums.npy", numArray)

# Decoding th image by getting some needed values.
lSize, rSize, tSize, bSize, rows, cols, bestStartX, bestEndX, bestStartY, bestEndY = numArray[0: numArray.size]

# Decoding Huffman
avgArr = decodeHuff(encodedArr, myDictionary)

# Decoding the averaging process.
image = decodeAverage(avgArr, lSize, rSize, tSize, bSize, rows, cols, bestStartX, bestEndX, bestStartY, bestEndY, 1)
cv2.imwrite("Decoded Image.png", image)

cv2.imshow('OS20', image)
cv2.waitKey(0)

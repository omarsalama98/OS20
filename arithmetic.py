import numpy as np


def encodeArithmetic(arr, blockSize):
    while arr.size % blockSize != 0:
        arr = np.append(arr, 0)

    unique, counts = np.unique(arr, return_counts=True)
    lengthDict = dict(zip(unique, counts))
    startDict = dict()
    accLength = 0

    for i in lengthDict:
        lengthDict[i] /= arr.size
        startDict[i] = accLength
        accLength += lengthDict[i]

    probabilitiesArr = np.zeros(256)

    for i in lengthDict:
        probabilitiesArr[i] = lengthDict[i]

    np.save("Probabilities Array", probabilitiesArr)
    encodedArr = np.zeros(int(arr.size / blockSize), np.float64)

    start = 0
    length = 1
    k = 0
    f = 0
    for i in arr:
        if f % blockSize == 0:
            if f != 0:
                encodedArr[k] = start + length / 2
                k += 1
            start = 0
            length = 1
        start += startDict[i] * length
        length = length * lengthDict[i]
        f += 1
    encodedArr[k] = start + length / 2

    np.save("Encoded Image", encodedArr)
    print(encodedArr)
    return encodedArr, probabilitiesArr


def decodeArithmetic(encodedArr, probabilitiesArr, blockSize, imgSize):
    decodedArr = np.zeros(imgSize, int)
    lengthDict = dict()
    startDict = dict()
    for i in range(0, probabilitiesArr.size):
        if probabilitiesArr[i] != 0:
            lengthDict[i] = probabilitiesArr[i]

    accLength = 0
    for i in lengthDict:
        startDict[i] = accLength
        accLength += lengthDict[i]

    start = 0
    length = 1
    k = 0
    for i in encodedArr:
        for f in range(0, blockSize):
            for j in startDict:
                if i < ((lengthDict[j] + startDict[j]) * length + start):
                    decodedArr[k] = j
                    k += 1
                    start += startDict[j] * length
                    length = length * lengthDict[j]
                    break
        start = 0
        length = 1

    return decodedArr

# To Think about:

# Why not make the block size dynamic
# As long as the precision of float isn't lost we continue
# and then add the block size number to the beginning of each block
# To avoid adding much which would result in a negative effect we could have two values
# for block sizes and indicate which one with one bit 0 or 1.


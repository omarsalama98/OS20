import numpy as np
bl_h = 2
bl_w = 2


def calcAvg(block):
    sum = 0
    smlArr = np.array([0], int)
    smlArr = np.delete(smlArr, 0)
    for p in block:
        for k in p:
            smlArr = np.append(smlArr, k)

    maxDiff = 0
    maxDiffIndex = 0
    for i in range(smlArr.size):
        diff = 0
        for j in range(smlArr.size):
            diff += abs(smlArr[j] - smlArr[i])
        if diff > maxDiff:
            maxDiff = diff
            maxDiffIndex = i
    smlArr = np.delete(smlArr, maxDiffIndex)
    for i in smlArr:
        sum += i
    sum /= smlArr.size
    return sum.__int__()


def encodeAvg(img):
    rows, cols = img.shape
    myArr = np.zeros(((rows*cols)/4).__int__(), np.uint8)
    index = 0
    for row in np.arange(rows - bl_h + 1, step=bl_h):
        for col in np.arange(cols - bl_w + 1, step=bl_w):
            if index == myArr.size:
                break
            myArr[index] = calcAvg(img[row:row + bl_h, col:col + bl_w])
            index += 1
    return myArr


def decodeAvg(img, myArr):
    rows, cols = img.shape
    index = 0
    for row in np.arange(rows - bl_h + 1, step=bl_h):
        for col in np.arange(cols - bl_w + 1, step=bl_w):
            if index == myArr.size:
                break
            img[row:row + bl_h, col:col + bl_w] = myArr[index]
            index += 1


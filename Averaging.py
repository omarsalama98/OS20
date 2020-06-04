import numpy as np


def encodeAverage(img, leftBound, rightBound, upBound, lowBound, hQ):
    rows, cols = img.shape

    avgArrLeft = encodeToThird(img[0: rows, 0: leftBound])
    avgArrRight = encodeToThird(img[0: rows, rightBound: cols])
    avgArrTop = encodeToThird(img[0: upBound, leftBound: rightBound])
    avgArrBottom = encodeToThird(img[lowBound: rows, leftBound: rightBound])

    if hQ:
        avgArrCenter = (img[upBound: lowBound, leftBound: rightBound]).flatten()
    else:
        avgArrCenter = encodeHalf(img[upBound: lowBound, leftBound: rightBound])

    avgArr = np.zeros(1, np.uint8)
    avgArr = np.delete(avgArr, 0)
    avgArr = np.append(avgArr, avgArrLeft)
    avgArr = np.append(avgArr, avgArrRight)
    avgArr = np.append(avgArr, avgArrTop)
    avgArr = np.append(avgArr, avgArrBottom)
    avgArr = np.append(avgArr, avgArrCenter)
    return avgArr, avgArrLeft.size, avgArrRight.size, avgArrTop.size, avgArrBottom.size


def decodeAverage(avgArr, avgArrLeftSize, avgArrRightSize, avgArrTopSize, avgArrBottomSize, rows, cols, leftBound,
                  rightBound, upBound, lowBound, hQ):
    img = np.zeros(rows * cols, np.uint8)
    img = np.reshape(img, (rows, -1))
    avgArrLeft = np.zeros(avgArrLeftSize, np.uint8)
    avgArrRight = np.zeros(avgArrRightSize, np.uint8)
    avgArrTop = np.zeros(avgArrTopSize, np.uint8)
    avgArrBottom = np.zeros(avgArrBottomSize, np.uint8)
    avgArrCenter = np.zeros(avgArr.size - (avgArrLeftSize + avgArrRightSize + avgArrTopSize + avgArrBottomSize),
                            np.uint8)
    lI, rI, tI, bI, cI = 0, 0, 0, 0, 0
    for i in range(avgArr.size):
        if i < avgArrLeftSize:
            avgArrLeft[lI] = avgArr[i]
            lI += 1
        elif i < avgArrLeftSize + avgArrRightSize:
            avgArrRight[rI] = avgArr[i]
            rI += 1
        elif i < avgArrLeftSize + avgArrRightSize + avgArrTopSize:
            avgArrTop[tI] = avgArr[i]
            tI += 1
        elif i < avgArrLeftSize + avgArrRightSize + avgArrTopSize + avgArrBottomSize:
            avgArrBottom[bI] = avgArr[i]
            bI += 1
        else:
            avgArrCenter[cI] = avgArr[i]
            cI += 1

    arrLeft = decodeThird(avgArrLeft, rows, leftBound)
    arrRight = decodeThird(avgArrRight, rows, (cols - rightBound))
    arrTop = decodeThird(avgArrTop, upBound, (rightBound - leftBound))
    arrBottom = decodeThird(avgArrBottom, (rows - lowBound), (rightBound - leftBound))

    if hQ:
        k = 0
        arrCenter = np.zeros((lowBound - upBound) * (rightBound - leftBound), np.uint8)
        arrCenter = np.reshape(arrCenter, ((lowBound - upBound), -1))
        for i in range((lowBound - upBound)):
            for j in range((rightBound - leftBound)):
                if k == avgArrCenter.size:
                    break
                arrCenter[i, j] = avgArrCenter[k]
                k += 1
    else:
        arrCenter = decodeHalf(avgArrCenter, (lowBound - upBound), (rightBound - leftBound))

    for i in range(rows):
        for j in range(leftBound):
            img[i, j] = arrLeft[i, j]

    for i in range(rows):
        for j in range((cols - rightBound)):
            img[i, j + rightBound] = arrRight[i, j]

    for i in range(upBound):
        for j in range(rightBound - leftBound):
            img[i, j + leftBound] = arrTop[i, j]

    for i in range((rows - lowBound)):
        for j in range(rightBound - leftBound):
            img[i + lowBound, j + leftBound] = arrBottom[i, j]

    for i in range(lowBound - upBound):
        for j in range(rightBound - leftBound):
            img[i + upBound, j + leftBound] = arrCenter[i, j]

    return img


def encodeHalf(img):
    rows, cols = img.shape
    encodedArr = np.array(1, np.uint8)
    encodedArr = np.delete(encodedArr, 0)
    i = 0
    while i < rows:
        j = 0
        while j < cols:
            encodedArr = np.append(encodedArr, img[i, j])
            j += 1 if i % 3 == 0 else 2
        i += 1
    return encodedArr


def decodeHalf(encodedArr, rows, cols):
    zerosImage = np.zeros((rows * cols), dtype=np.uint8)
    zerosImage = np.reshape(zerosImage, (rows, -1))
    k = 0
    i = 0
    while i < rows:
        j = 0
        while j < cols:
            zerosImage[i, j] = encodedArr[k]
            j += 1 if i % 3 == 0 else 2
            k += 1
        i += 1
    for i in range(rows):
        if i % 3 == 0:
            continue
        for j in range(cols):
            if j % 2 == 0:
                continue
            sum = 0
            num = 0
            if j != cols - 1 and zerosImage[i, j + 1] != 0:
                sum += zerosImage[i, j + 1] * 3
                num += 3
            if zerosImage[i - 1, j] != 0:
                sum += zerosImage[i - 1, j] * 3
                num += 3
            if zerosImage[i, j - 1] != 0:
                sum += zerosImage[i, j - 1] * 3
                num += 3
            if i != rows - 1 and zerosImage[i + 1, j] != 0:
                sum += zerosImage[i + 1, j] * 3
                num += 3

            sum += zerosImage[i - 1, j - 1]
            num += 1
            if j != cols - 1:
                sum += zerosImage[i - 1, j + 1]
                num += 1
            if i != rows - 1 and zerosImage[i + 1, j - 1] != 0:
                sum += zerosImage[i + 1, j - 1]
                num += 1
            if i != rows - 1 and j != cols - 1 and zerosImage[i + 1, j + 1] != 0:
                sum += zerosImage[i + 1, j + 1]
                num += 1
            zerosImage[i, j] = sum / num
    return zerosImage


def encodeAvgTwos(img):
    img = np.uint16(img)
    rows, cols = img.shape
    avgArr = np.zeros((img.size / 2).__int__(), np.uint8)
    k = 0
    for i in range(rows):
        j = 0
        while j < (cols - 1):
            if k == avgArr.size:
                print(i, j, rows, cols)
                break
            if 1 < j < cols - 2:
                avgArr[k] = (img[i, j] + img[i, j + 1] + img[i, j - 1] + img[i, j + 2]) / 4
            else:
                avgArr[k] = (img[i, j] + img[i, j + 1]) / 2
            k += 1
            j += 2
    return avgArr


def decodeAvgTwos(avgArr, rows, cols):
    img = np.zeros(rows * cols, np.uint8)
    i = 0
    j = 0
    while j < avgArr.size and i < img.size:
        img[i] = avgArr[j]
        img[i + 1] = avgArr[j]
        i += 2
        j += 1
    img = np.reshape(img, (rows, -1))
    return img


def calcAvgEvenBlock(block):
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


def encodeAvgEvenBlock(img, bl_w, bl_h):
    rows, cols = img.shape
    myArr = np.zeros(((rows * cols) / (bl_h * bl_w)).__int__(), np.uint8)
    index = 0
    for row in np.arange(rows - bl_h + 1, step=bl_h):
        for col in np.arange((cols - bl_w + 1), step=bl_w):
            if index == myArr.size:
                break
            myArr[index] = calcAvgEvenBlock(img[row:row + bl_h, col:col + bl_w])
            index += 1
    return myArr


def decodeAvgEvenBlock(img, myArr, bl_w, bl_h):
    rows, cols = img.shape
    index = 0
    for row in np.arange(rows - bl_h + 1, step=bl_h):
        for col in np.arange(cols - bl_w + 1, step=bl_w):
            if index == myArr.size:
                break
            img[row:row + bl_h, col:col + bl_w] = myArr[index]
            index += 1


def calcAvgOddBlock(block, bl_w, bl_h):
    sum1 = 0
    sum2 = 0
    for p in range(bl_h):
        for k in range(bl_w):
            if p + k < bl_w:
                sum1 += block[p, k]
            else:
                sum2 += block[p, k]

    sum1 /= ((bl_h * bl_w) / 2)
    sum2 /= ((bl_h * bl_w) / 2)
    return sum1.__int__(), sum2.__int__()


def encodeAvgOddBlock(img, bl_w, bl_h):
    rows, cols = img.shape
    myArr = np.zeros(((rows * cols) / (bl_h * bl_w) * 2).__int__(), np.uint8)
    index = 0
    for row in np.arange(rows - bl_h + 1, step=bl_h):
        for col in np.arange(cols - bl_w + 1, step=bl_w):
            if index == myArr.size:
                break
            myArr[index], myArr[index + 1] = calcAvgOddBlock(img[row:row + bl_h, col:col + bl_w], bl_w, bl_h)
            index += 2
    return myArr


def decodeAvgOddBlock(img, myArr, bl_w, bl_h):
    rows, cols = img.shape
    index = 0
    for row in np.arange(rows - bl_h + 1, step=bl_h):
        for col in np.arange(cols - bl_w + 1, step=bl_w):
            if index == myArr.size:
                break
            for p in range(bl_h):
                for k in range(bl_w):
                    if p + k < bl_w:
                        img[row + p, col + k] = myArr[index]
                    else:
                        img[row + p, col + k] = myArr[index + 1]
            index += 2


def encodeToThird(img):
    rows, cols = img.shape
    encodedArr = np.array(1, np.uint8)
    encodedArr = np.delete(encodedArr, 0)
    i = rows - 1
    while i >= 0:
        j = 0
        k = i
        while j < cols and k < rows:
            encodedArr = np.append(encodedArr, img[k, j])
            k += 1
            j += 1
        i -= 3
    j = 1
    while j < cols:
        i = 0
        k = j
        while k < cols and i < rows:
            encodedArr = np.append(encodedArr, img[i, k])
            i += 1
            k += 1
        j += 3

    return encodedArr


def decodeThird(encodedArr, rows, cols):
    zerosImage = np.zeros((rows * cols), dtype=np.uint8)
    zerosImage = np.reshape(zerosImage, (rows, -1))
    i = rows - 1
    k = 0
    while i >= 0:
        w = i
        j = 0
        while j < cols and w < rows:
            zerosImage[w, j] = encodedArr[k]
            w += 1
            j += 1
            k += 1
        i -= 3
    j = 1
    while j < cols:
        w = j
        i = 0
        while w < cols and i < rows:
            zerosImage[i, w] = encodedArr[k]
            i += 1
            w += 1
            k += 1
        j += 3

    i = 1
    while i < rows:
        j = 0
        while j < cols - 1:
            if zerosImage[i, j] == 0 and zerosImage[i - 1, j] != 0 and zerosImage[i, j + 1] != 0:
                x = int(zerosImage[i - 1, j])
                x += zerosImage[i, j + 1]
                x /= 2
                zerosImage[i, j] = x
            j += 1
        i += 1

    for i in range(rows):
        for j in range(cols):
            sum = 0
            num = 0
            if j != cols - 1 and zerosImage[i, j + 1] != 0:
                sum += zerosImage[i, j + 1]
                num += 1
            if i != 0 and zerosImage[i - 1, j] != 0:
                sum += zerosImage[i - 1, j]
                num += 1
            if j != 0 and zerosImage[i, j - 1] != 0:
                sum += zerosImage[i, j - 1]
                num += 1
            if i != rows - 1 and zerosImage[i + 1, j] != 0:
                sum += zerosImage[i + 1, j]
                num += 1
            zerosImage[i, j] = sum / num

    return zerosImage

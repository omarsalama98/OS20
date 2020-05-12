from scipy.fftpack import dct, idct
import numpy as np


# implement 2D DCT
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')


# implement 2D IDCT
def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')


def applyDCT(img):
    img = np.uint8(img)
    block_img = np.zeros(img.shape, dtype=np.int16)
    im_h, im_w = img.shape[:2]
    for i in range(im_h):
        for j in range(im_w):
            img[i, j] -= 128  # as dct works on the range -128 -> 127
    bl_h, bl_w = 5, 5  # 8x8 Blocks

    for row in np.arange(im_h - bl_h + 1, step=bl_h):
        for col in np.arange(im_w - bl_w + 1, step=bl_w):
            block_img[row:row + bl_h, col:col + bl_w] = dct2(img[row:row + bl_h, col:col + bl_w])
    print(block_img, block_img.size, block_img.shape)
    return block_img


def applyIDCT(arr, img):
    im_h, im_w = img.shape[:2]
    bl_h, bl_w = 5, 5  # 8x8 Blocks

    for row in np.arange(im_h - bl_h + 1, step=bl_h):
        for col in np.arange(im_w - bl_w + 1, step=bl_w):
            img[row:row + bl_h, col:col + bl_w] = idct2(arr[row:row + bl_h, col:col + bl_w])

    for i in range(im_h):
        for j in range(im_w):
            img[i, j] += 128  # to return it to its initial value
    return img


def removeZerosAfterDCT(arr, im_h, im_w):
    bl_h, bl_w = 5, 5  # 8x8 Blocks
    encodedArr = np.zeros(1, np.int16)
    encodedArr = np.delete(encodedArr, 0)
    for row in np.arange(im_h - bl_h + 1, step=bl_h):
        for col in np.arange(im_w - bl_w + 1, step=bl_w):
            rows = 5
            columns = 5
            solution = [[] for i in range(rows + columns - 1)]
            for i in range(rows):
                for j in range(columns):
                    sum = i + j
                    if sum % 2 == 0:
                        # add at beginning
                        solution[sum].insert(0, arr[row:row + bl_h, col:col + bl_w][i][j])
                    else:
                        # add at end of the list
                        solution[sum].append(arr[row:row + bl_h, col:col + bl_w][i][j])
            brk = False
            for i in solution:
                for j in range(i.__len__()):
                    encodedArr = np.append(encodedArr, i[j])
                    if j == 3:
                        brk = True
                if brk:
                    break

    print(encodedArr.size)
    return encodedArr


def addZerosForIDCT(arr):
    decodedArr = np.zeros(1, np.int16)
    for i in range(arr.size):
        if i != 0 and i % 10 == 0:
            decodedArr = np.append(decodedArr, np.zeros(15, np.int16))
        decodedArr = np.append(decodedArr, arr[i])
    decodedArr = np.append(decodedArr, np.zeros(15, np.int16))
    decodedBlocks = decodedArr.reshape((305, 380))
    print(decodedBlocks, decodedBlocks.size, decodedBlocks.shape)
    return decodedBlocks

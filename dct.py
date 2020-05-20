from scipy.fftpack import dct, idct
import numpy as np

blockDim = 8
bl_h = blockDim
bl_w = blockDim

quantizationMatrix = [[16, 11, 10, 16, 24, 40, 51, 61],
                      [12, 12, 14, 19, 26, 58, 60, 55],
                      [14, 13, 16, 24, 40, 57, 69, 56],
                      [14, 17, 22, 29, 51, 87, 80, 62],
                      [18, 22, 37, 56, 68, 109, 103, 77],
                      [24, 35, 55, 64, 81, 104, 103, 92],
                      [49, 64, 78, 77, 103, 121, 120, 101],
                      [72, 92, 95, 98, 112, 100, 103, 99]]
# quantizationMatrix = np.ones((8, 8))


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

    for row in np.arange(im_h - bl_h + 1, step=bl_h):
        for col in np.arange(im_w - bl_w + 1, step=bl_w):
            block_img[row:row + bl_h, col:col + bl_w] = dct2(img[row:row + bl_h, col:col + bl_w])

    return block_img


def applyIDCT(arr, img):
    im_h, im_w = img.shape[:2]
    for row in np.arange(im_h - bl_h + 1, step=bl_h):
        for col in np.arange(im_w - bl_w + 1, step=bl_w):
            img[row:row + bl_h, col:col + bl_w] = idct2(arr[row:row + bl_h, col:col + bl_w])

    for i in range(im_h):
        for j in range(im_w):
            img[i, j] += 128  # to return it to its initial value
    return img


def removeZerosAfterDCT(arr, im_h, im_w):
    encodedArr = np.zeros(1, np.int16)
    encodedArr = np.delete(encodedArr, 0)
    for row in np.arange(im_h - bl_h + 1, step=bl_h):
        for col in np.arange(im_w - bl_w + 1, step=bl_w):
            for i in range(blockDim):
                j = 0
                while j <= blockDim - 1 - i:
                    encodedArr = np.append(encodedArr, arr[row:row + bl_h, col:col + bl_w][i][j])
                    j += 1

    return encodedArr


def addZerosForIDCT(arr, img):
    decodedArr = np.zeros(img.shape, dtype=np.int16)
    im_h, im_w = img.shape
    k = 0
    for row in np.arange(im_h - bl_h + 1, step=bl_h):
        for col in np.arange(im_w - bl_w + 1, step=bl_w):
            for i in range(blockDim):
                j = 0
                while j <= blockDim - 1 - i:
                    decodedArr[row:row + bl_h, col:col + bl_w][i][j] = arr[k]
                    k += 1
                    j += 1

    return decodedArr
# j = 0
# while j <= blockDim - 1 - i:
#    decodedArr[row:row + bl_h, col:col + bl_w][i][j] = arr[k]
#    k += 1
#    j += 1

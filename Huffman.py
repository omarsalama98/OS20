from queue import PriorityQueue
import numpy as np

class node:

    def __lt__(self, other):
        return self.value < other.value
    rightChild = None
    leftChild = None
    key = 0
    value = 0
    code = ""


def encodeHuff(arr):

    myDictionary = {}
    for i in arr:
        if i not in myDictionary:
            myDictionary[i] = 1
        else:
            myDictionary[i] += 1

    myQueue = PriorityQueue()

    list = [(v, k) for k, v in myDictionary.items()]

    for i in list:
        x = node()
        x.key = i[1]
        x.value = i[0]
        myQueue.put(x)

    root = node()

    while myQueue.qsize() > 1:
        x = myQueue.get()
        y = myQueue.get()
        z = node()
        z.leftChild = x
        z.rightChild = y
        z.value = x.value + y.value
        z.key = '0'
        root = z
        myQueue.put(z)

    # Adds Codes to built tree starting from its root
    def putCodes(node):

        if node.leftChild is not None:
            node.leftChild.code = node.code + "0"
            putCodes(node.leftChild)

        if node.rightChild is not None:
            node.rightChild.code = node.code + "1"
            putCodes(node.rightChild)

        else:
            return

    # Searches through tree for a given key (Starting from root)
    def getCode(mNode, key):

        if mNode.leftChild is not None:
            s = getCode(mNode.leftChild, key)
            if s is not None:
                return s

        if mNode.rightChild is not None:
            j = getCode(mNode.rightChild, key)
            if j is not None:
                return j

        else:
            if key == mNode.key:
                return mNode.code
            else:
                return

    putCodes(root)

    sum = 0
    for i in myDictionary:
        sum += (myDictionary[i] * getCode(root, i).__len__())
    sum /= 8
    index = 0
    myStrArr = [""]*arr.size
    for i in arr:
        myStrArr[index] = (getCode(root, i))
        index += 1
    codesString = "".join(myStrArr)
    i = 0
    encodedArr = np.array([0], np.uint8)
    encodedArr = np.delete(encodedArr, 0)
    while i < (codesString.__len__() - 8):
        x = int(codesString[i: i + 8], 2)
        encodedArr = np.append(encodedArr, np.uint8(x))
        i += 8
    x = int(codesString[i: codesString.__len__()], 2)
    encodedArr = np.append(encodedArr, np.uint8(x))
    print(encodedArr.size)

    return encodedArr, myDictionary


def decodeHuff(encodedArr, myDictionary):

    codesStringArr = [""] * encodedArr.size
    index = 0
    for i in encodedArr:
        num = bin(i).replace("0b", "")
        rNum = ""
        s = 8 - num.__len__()
        while s > 0:
            rNum += "0"
            s -= 1
        codesStringArr[index] = rNum + num
        index += 1
    codesString = "".join(codesStringArr)
    myQueue = PriorityQueue()

    list = [(v, k) for k, v in myDictionary.items()]

    for i in list:
        x = node()
        x.key = i[1]
        x.value = i[0]
        myQueue.put(x)

    root = node()

    while myQueue.qsize() > 1:
        x = myQueue.get()
        y = myQueue.get()
        z = node()
        z.leftChild = x
        z.rightChild = y
        z.value = x.value + y.value
        z.key = '0'
        root = z
        myQueue.put(z)

    # Adds Codes to built tree starting from its root
    def putCodes(mNode):

        if mNode.leftChild is not None:
            mNode.leftChild.code = mNode.code + "0"
            putCodes(mNode.leftChild)

        if mNode.rightChild is not None:
            mNode.rightChild.code = mNode.code + "1"
            putCodes(mNode.rightChild)

        else:
            return

    def getKeys(mRoot, code):
        decodedArr = np.array([0], np.uint8)
        decodedArr = np.delete(decodedArr, 0)
        ind = 0
        while ind < code.__len__():
            mNode = mRoot
            while ind < code.__len__():
                if code[ind] == "0":
                    if mNode.leftChild is None:
                        decodedArr = np.append(decodedArr, mNode.key)
                        break
                    else:
                        mNode = mNode.leftChild
                else:
                    if mNode.rightChild is None:
                        decodedArr = np.append(decodedArr, mNode.key)
                        break
                    else:
                        mNode = mNode.rightChild
                ind += 1
        return decodedArr

    putCodes(root)
    arr = getKeys(root, codesString)
    return arr


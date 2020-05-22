import cv2
import numpy as np


def encodeLZ(arr, slidingWindSize, lookUpBuffSize):

    lookUpBufferIndex = slidingWindSize - lookUpBuffSize
    searchBufferIndex = 0

    searchBuffer = np.array(arr[searchBufferIndex: lookUpBufferIndex: 1])
    lookUpBuffer = np.array(arr[lookUpBufferIndex: lookUpBufferIndex + lookUpBuffSize: 1])

    print("Encoding Please wait....")

    codesMatchingArray = [(0, 0)]
    codesValuesArray = np.array([0], np.uint8)
    codesMatchingArray.remove((0, 0))
    codesValuesArray = np.delete(codesValuesArray, 0)
    while searchBufferIndex + searchBuffer.size < arr.size:
        j = 0
        searchBuffer = np.array(arr[searchBufferIndex: lookUpBufferIndex: 1])
        lookUpBuffer = np.array(arr[lookUpBufferIndex: lookUpBufferIndex + lookUpBuffSize: 1])
        matchingIndex = 0
        matchingLength = 0
        bestMatchingCode = (matchingIndex, matchingLength)
        bestValuesCode = arr[lookUpBufferIndex]
        for i in range(0, searchBuffer.size):
            if j >= lookUpBuffer.size:
                bestMatchingCode = (matchingIndex, matchingLength)
                bestValuesCode = arr[
                    lookUpBufferIndex + matchingLength] if lookUpBufferIndex + matchingLength < arr.size else -1
                break
            if searchBuffer[i] == lookUpBuffer[j]:
                matchingLength += 1
                j += 1
                if matchingLength == 1:
                    matchingIndex = searchBuffer.size - i
                if i == searchBuffer.size - 1:
                    k = lookUpBufferIndex
                    while k < arr.size and j < lookUpBuffSize and arr[k] == lookUpBuffer[j]:
                        matchingLength += 1
                        j += 1
                        k += 1
                    bestMatchingCode = (matchingIndex, matchingLength)
                    if j < lookUpBuffer.size:
                        bestValuesCode = lookUpBuffer[j]
                    else:
                        bestValuesCode = -1
            else:
                if matchingLength >= bestMatchingCode[1]:
                    bestMatchingCode = (matchingIndex, matchingLength)
                    bestValuesCode = arr[lookUpBufferIndex + matchingLength]
                matchingLength = 0
                matchingIndex = 0
                j = 0
        codesMatchingArray.append(bestMatchingCode)
        codesValuesArray = np.append(codesValuesArray, bestValuesCode)
        matchingLength = bestMatchingCode[1]
        searchBufferIndex += matchingLength + 1
        lookUpBufferIndex += matchingLength + 1

    np.save("Encoded Image's Matching Index and Length", codesMatchingArray)
    np.save("Encoded Image's Pixel Values", codesValuesArray)
    print("Encoding Finished.")
    return codesValuesArray, codesMatchingArray


def decodeLZ(searchBuffer, codesValuesArray, codesMatchingArray, slidingWindSize, lookUpBuffSize):
    print("Decoding Please wait....")
    lookUpBufferIndex = slidingWindSize - lookUpBuffSize
    searchBufferIndex = 0

    decodedArray = np.array(searchBuffer)
    for i in range(0, codesValuesArray.size):
        searchBuffer = np.array(decodedArray[searchBufferIndex: lookUpBufferIndex: 1])
        matchingLength = codesMatchingArray[i][1]
        if matchingLength == 0:
            decodedArray = np.append(decodedArray, codesValuesArray[i])
        else:
            matchingIndex = codesMatchingArray[i][0]
            matchingIndex = searchBuffer.size - matchingIndex
            overFlowLookUpIndex = decodedArray.size
            for j in range(0, matchingLength):
                if matchingIndex >= searchBuffer.size:
                    decodedArray = np.append(decodedArray, decodedArray[overFlowLookUpIndex])
                    overFlowLookUpIndex += 1
                else:
                    decodedArray = np.append(decodedArray, searchBuffer[matchingIndex])
                    matchingIndex += 1
            decodedArray = np.append(decodedArray, codesValuesArray[i])
        searchBufferIndex += matchingLength + 1
        lookUpBufferIndex += matchingLength + 1
    return decodedArray



import argparse

import cv2
import numpy as np


def applyDL(image):
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=True,
                    help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True,
                    help="path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, default=0.05,
                    help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    rows, cols = image.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843,
                                 (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()

    bestStartX, bestStartY, bestEndX, bestEndY = 0, 0, 0, 0

    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([cols, rows, cols, rows])
            (startX, startY, endX, endY) = box.astype("int")
            # display the prediction
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            print("[INFO] {}".format(label))
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          255, 2)
            if (endX - startX) * (endY - startY) > (bestEndX - bestStartX) * (bestEndY - bestStartY):
                bestStartX = startX
                bestStartY = startY
                bestEndX = endX
                bestEndY = endY
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)

    # cv2.imshow("Output", image)

    while bestStartX % 10 != 0:
        bestStartX -= 1
    while bestEndX % 10 != 0:
        bestEndX += 1
    while bestEndY % 10 != 0:
        bestEndY += 1
    while bestStartY % 10 != 0:
        bestStartY -= 1

    return bestStartX, bestStartY, bestEndX, bestEndY

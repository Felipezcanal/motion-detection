from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import argparse
from pprint import pprint
import pickle
import imutils
import time


currentBackground = 0
n = 0
lastsrc = 0
lasts = []
lastpasstime = 0
lastpasspos = 0
personCounter = 0
out = 0

detectionRectangle = [(415, 525), (590, 570)]


video_capture = cv.VideoCapture("lab.mp4")



def thresh_callback(val, src_gray, original):
    global lastpasstime, detectionRectangle, personCounter, lastpasspos




    original2 = original.copy()
    src_gray = cv.blur(src_gray, (3,3))
    canny = cv.Canny(src_gray, 50, 20)
    contours, _ = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    thresh = cv.threshold(src_gray, 25, 255, cv.THRESH_BINARY)[1]

    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv.dilate(thresh, None, iterations=18)

    # canny = cv.Canny(thresh, 50, 20)
    contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(contours)



    cv.rectangle(original2, detectionRectangle[0], detectionRectangle[1], (0, 0, 255), 2)
    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv.contourArea(c) < 40000:
            continue

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv.boundingRect(c)
        cv.rectangle(original2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.circle(original2, (int(x+(w/2)), y+h), 2, (255, 0, 0), 2)

        logic = int(x+(w/2)) > detectionRectangle[0][0] and int(x+(w/2)) < detectionRectangle[1][0] and (y+h) > detectionRectangle[0][1] and (y+h) < detectionRectangle[1][1]
        if logic:

            entrando = 0

            cv.rectangle(original2, detectionRectangle[0], detectionRectangle[1], (255, 255, 255), 2)
            if(time.time() - lastpasstime < 2.2):
                lastpasstime = time.time()
                continue

            if((y+h) > lastpasspos):
                entrando = 1
            pprint(time.time() - lastpasstime)
            pprint(y+h)

            lastpasstime = time.time()
            if((y+h) > (detectionRectangle[1][1] - ((detectionRectangle[1][1] - detectionRectangle[0][1])/2) ) ):
            # if entrando == 0:
                personCounter -= 1
            else:
                personCounter += 1

    cv.putText(original2, "Contador: ", (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), lineType=cv.LINE_AA)
    cv.putText(original2, str(personCounter), (180, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), lineType=cv.LINE_AA)




    # for i in range(len(contours)):
    #
    #
    #     c = contours[i]
    #     extLeft = tuple(c[c[:, :, 0].argmin()][0])
    #     extRight = tuple(c[c[:, :, 0].argmax()][0])
    #     extTop = tuple(c[c[:, :, 1].argmin()][0])
    #     extBot = tuple(c[c[:, :, 1].argmax()][0])
    #
    #     topLeft = (extLeft[0], extTop[1])
    #     botRight = (extRight[0], extBot[1])
    #
    #     if (len(src_gray[topLeft[0]:botRight[0], topLeft[1]:botRight[1]]) > 0):
    #         cv.rectangle(original2,topLeft,botRight,(0,255,0),3)



            # while True:
            #     key = cv.waitKey(1) & 0xFF
            #     if key == ord("p"):
            #         saveFingerPrint(img)
            #         pprint("salvo")
            #         break
            #     elif key == ord("a"):
            #         assess(img)
            #     elif key != 255:
            #         break



    # cv.namedWindow("Resultado", cv.WINDOW_KEEPRATIO)
    cv.namedWindow("Thresh", cv.WINDOW_KEEPRATIO)
    cv.imshow('Resultado', original2)
    cv.imshow('Thresh', thresh)
    out.write(original2)
    cv.waitKey(5)



def readVideo():
    global currentBackground, n, lastsrc, out
    for i in range(1):
        ret, src = video_capture.read()

    x, y, _ = src.shape
    out = cv.VideoWriter('outpy.mp4', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 24, (y, x))
    icount = 0
    while True:
        # pprint(icount)
        icount += 1
        if not video_capture.isOpened():
            print('Unable to load camera.')
            sleep(5)
            pass

        # Capture frame-by-frame
        ret, src = video_capture.read()

        if(n >0):
            currentBackground2 = currentBackground.astype("float64")
            src2 = src.astype("float64")
            if n < 10:
                n += 1
            else:
                del lasts[0]

            # pprint(len(lasts))
            if(n > 3):
                currentBackground2 = lasts[0]
                for i in range(n-3):
                    # currentBackground2 = cv.addWeighted(currentBackground2, 1/2, lasts[n-3-i], 1/2, 0)
                    a = i+1
                    b = 1
                    div = a+b
                    currentBackground2 = cv.addWeighted(currentBackground2, a/div, lasts[i+1], b/div, 0)
                    # pprint(i)
                # print("\n")

            # newcurrent = cv.addWeighted(currentBackground2, (n-1)/n, src2, 1/n, 0)

            # pprint(currentBackground2[0][0])
            # pprint(src2[0][0])
            # pprint(newcurrent[0][0])
            # print("\n")
            #     cv.imshow("asd", currentBackground2.astype("uint8"))
            # newcurrent = cv.divide(newcurrent, n)
                currentBackground = currentBackground2.astype("uint8")
            lasts.append(src2)
        else:
            currentBackground = src
            n+= 1

        src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

        teste = currentBackground.copy()
        sub = cv.absdiff(src, currentBackground)
        # sub = src - currentBackground

        src_gray = cv.cvtColor(sub, cv.COLOR_BGR2GRAY)
        # cv.imshow("Subtraction", src_gray)
        cv.waitKey(5)

        lastsrc = src

        # thresh_callback(100, src_gray, src)
        thresh_callback(100, src_gray, src)



readVideo()

cv.namedWindow("Source", cv.WINDOW_KEEPRATIO)
cv.namedWindow("Subtraction", cv.WINDOW_KEEPRATIO)



while True:
    key = cv.waitKey(1) & 0xFF
    if key == ord("c"):
        break

cv.destroyAllWindows()

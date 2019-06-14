from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import argparse
from pprint import pprint
import pickle


currentBackground = cv.imread("frame2946.bmp");
n = 0
lastsrc = 0


video_capture = cv.VideoCapture("road.mp4")



def thresh_callback(val, src_gray, original):
    original2 = original.copy()
    contours, _ = cv.findContours(src_gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


    # Get the moments
    # mu = [None]*len(contours)
    # for i in range(len(contours)):
    #     mu[i] = cv.moments(contours[i])

    # Get the mass centers
    # mc = [None]*len(contours)
    # for i in range(len(contours)):
    #     mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))


    # Draw contours

    # drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    #
    # for i in range(len(contours)):
    #     color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    #     cv.drawContours(drawing, contours, i, color, 2)
    #     cv.circle(drawing, (int(mc[i][0]), int(mc[i][1])), 4, color, -1)


    # Calculate the area with the moments 00 and compare with the result of the OpenCV function
    for i in range(len(contours)):
        if(len(contours[i]) < 150):
            continue
        else:
            # cv.imshow('Contours'+str(i), )
            # minx = max(contours[i])
            # pprint(max(contours[i][0]))
            c = contours[i]
            extLeft = tuple(c[c[:, :, 0].argmin()][0])
            extRight = tuple(c[c[:, :, 0].argmax()][0])
            extTop = tuple(c[c[:, :, 1].argmin()][0])
            extBot = tuple(c[c[:, :, 1].argmax()][0])
            topLeft = (extLeft[0], extTop[1])
            botRight = (extRight[0], extBot[1])
            # cv.circle(drawing, topLeft, 8, (255, 255, 255), -1)
            # cv.circle(drawing, botRight, 8, (255, 255, 255), -1)

            # pprint(src_gray[topLeft[0]:botRight[0], topLeft[1]:botRight[1]])
            if (len(src_gray[topLeft[0]:botRight[0], topLeft[1]:botRight[1]]) > 0):
                cv.namedWindow("Contours2", cv.WINDOW_KEEPRATIO)
                img = src_gray[topLeft[1]:botRight[1] , topLeft[0]:botRight[0]]
                # cv.imshow('Contours3', img)
                # _, img = cv.threshold(resize(img, 512, 512), 100, 255, 0)
                # cv.imshow('Contours2', img)

                cv.rectangle(original2,topLeft,botRight,(0,255,0),3)



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




            # print(' * Contour[%d] - Area (M_00) = %.2f - Area OpenCV: %.2f - Length: %.2f' % (i, mu[i]['m00'], cv.contourArea(contours[i]), cv.arcLength(contours[i], True)))


    cv.namedWindow("Contours", cv.WINDOW_KEEPRATIO)
    cv.imshow('Contours', original2)
    cv.waitKey(5)




def readVideo():
    global currentBackground, n, lastsrc
    while True:
        if not video_capture.isOpened():
            print('Unable to load camera.')
            sleep(5)
            pass

        # Capture frame-by-frame
        ret, src = video_capture.read()

        if(n >0):
            currentBackground2 = currentBackground.astype("float64")
            src2 = lastsrc.astype("float64")
            if n < 100:
                n += 1
            # else:
                # break

            newcurrent = cv.addWeighted(currentBackground2, (n-1)/n, src2, 1/n, 0)

            # pprint(currentBackground2[0][0])
            # pprint(src2[0][0])
            # pprint(newcurrent[0][0])
            # print("\n")
            cv.imshow("asd", newcurrent.astype("uint8"))
            # newcurrent = cv.divide(newcurrent, n)
            currentBackground = newcurrent.astype("uint8")
        else:
            n+= 1

        src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        sub = cv.subtract(src, currentBackground)

        cv.imshow("Subtraction", sub)
        cv.waitKey(5)

        lastsrc = src

        src_gray = cv.cvtColor(sub, cv.COLOR_BGR2GRAY)
        thresh_callback(100, src_gray, src)



readVideo()

cv.namedWindow("Source", cv.WINDOW_KEEPRATIO)
cv.namedWindow("Subtraction", cv.WINDOW_KEEPRATIO)



while True:
    key = cv.waitKey(1) & 0xFF
    if key == ord("c"):
        break

cv.destroyAllWindows()

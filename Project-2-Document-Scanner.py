import cv2
import numpy as np

widthImg = 640
heightImg = 480
# --------------------------opening camera--------------------------------
vid2 = cv2.VideoCapture(0)   # 0 shows the address of camera
vid2.set(3, widthImg)   # 3 shows channel for screen width
vid2.set(4, heightImg)   # 4 shows height
vid2.set(10, 150)  # 10 shows brightness channel


def pre_processing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCorner = cv2.Canny(imgBlur, 200, 200)
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCorner, kernel, iterations=2)
    imgThresh = cv2.erode(imgDial, kernel, iterations=2)

    return imgThresh


def get_contours(test):
    contours, hierarchy = cv2.findContours(test, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    biggest = np.array([])
    maxArea = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            # cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 3)
    return biggest


def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    # print("add", add)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]

    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


def get_warp(img, biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    OutputImg = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    return OutputImg


while True:
    success, img = vid2.read()
    img = cv2.resize(img, (widthImg, heightImg))
    imgContour = img.copy()
    imgThres = pre_processing(img)
    biggest = get_contours(imgThres)
    print(biggest)

    if (biggest != []):
        WarpedImg = get_warp(img, biggest)
        cv2.imshow("Video", WarpedImg)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

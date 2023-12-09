import numpy as np
import cv2 as cv
cap = cv.VideoCapture('/home/changhuan/data/stream.mp4')
if not cap.isOpened():
    print('NO VIDEO')
    exit()
while True:
    # 逐帧捕获
    ret, frame = cap.read()
    # 如果正确读取帧，ret为True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # 我们在框架上的操作到这里
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret2, binary = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
    ##Gaussian = cv.GaussianBlur(binary, (5, 5), 0)  # 高斯滤波
    ##draw_img = Gaussian.copy()
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)
    for i in (contours):
        x, y, w, h = cv.boundingRect(i)
        area = w * h
        if area <= 500 and area >= 50:
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    # 显示结果帧e
    cv.imshow('frame', frame)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break
# 完成所有操作后，释放捕获器
cap.release()
cv.destroyAllWindows()

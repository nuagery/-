import cv2
import numpy as np
import math

def img_show(name, src):
    cv2.imshow(name, src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image = cv2.imread('/home/changhuan/data/two.jpg')
height, width = image.shape[:2]
height = float(height)
width = float(width)
r = width/height
height = int(height / 5)
img = cv2.resize(image,(int(height*r),height), interpolation = cv2.INTER_LINEAR)

# 在彩色图像的情况下，解码图像将以b g r顺序存储通道
grid_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
# 从RGB色彩空间转换到HSV色彩空间
grid_HSV = cv2.cvtColor(grid_RGB, cv2.COLOR_RGB2HSV)





# H、S、V范围一：
lower1 = np.array([0,150,150])
upper1 = np.array([10,255,255])
mask1 = cv2.inRange(grid_HSV, lower1, upper1)       # mask1 为二值图像
res1 = cv2.bitwise_and(grid_RGB, grid_RGB, mask=mask1)
 
# H、S、V范围二：
lower2 = np.array([156,150,150])
upper2 = np.array([160,255,255])
mask2 = cv2.inRange(grid_HSV, lower2, upper2)
res2 = cv2.bitwise_and(grid_RGB,grid_RGB, mask=mask2)
 
# 将两个二值图像结果 相加
mask3 = mask1 + mask2
    
# 结果显示
cv2.imshow("mask3", mask3)
cv2.imshow("grid_RGB", grid_RGB[:,:,::-1])           # imshow()函数传入的变量也要为b g r通道顺序
cv2.waitKey(0)
cv2.destroyAllWindows()

a = []
#imgray = cv2.cvtColor(mask3, cv2.COLOR_BGR2GRAY)
#img_show('2',imgray)
#gray = cv2.cvtColor(imgray, cv2.COLOR_BGR2GRAY)
ret2, binary = cv2.threshold(mask3, 150, 255, cv2.THRESH_BINARY)
Gaussian = cv2.GaussianBlur(binary, (5, 5), 0)  # 高斯滤波
draw_img = Gaussian.copy()
# 输出的第一个值为图像，第二个值为轮廓信息，第三个为层级信息
cnt,hierarchy = cv2.findContours(draw_img,cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
##cv2.drawContours(img, cnt, -1, (255,0,0), 3)
##img_show('3',img)


wid = []
hei = []
for i in (cnt):
    area = cv2.contourArea(i)
    if area >= 100:
        x, y, w, h = cv2.boundingRect(i)
        ar = w*h
        if h / w >= 1 and area*2 >= ar :
            wid.append(w)
            hei.append(h)
            a.append(i)
print(wid)
print(hei)
c = []
t = []
y_l = []
y_s = []
right = []
left = []
for j in range(0,len(a)):
    perimeter1 = cv2.arcLength(a[j],True)
    l = j + 1
    while l < (len(a)):
        perimeter2 = cv2.arcLength(a[l],True)
        per_ratio = perimeter1 / perimeter2
        if per_ratio <= 2 and per_ratio >= 0.5 :
            right.append(a[j])
            left.append(a[l])
            l = l + 1
        else:
            l = l + 1
for j in (right):
    for o in (left):
        (cx_1,cy_1), (l_1,w_1), theta_1 = cv2.minAreaRect(j)
        (cx_2,cy_2), (l_2,w_2), theta_2 = cv2.minAreaRect(o)
        rect_1 = (cx_1,cy_1), (l_1,w_1), theta_1
        rect_2 = (cx_2,cy_2), (l_2,w_2), theta_2
        print('jiao',theta_1,theta_2)
        print('point',(cx_1,cy_1),(cx_2,cy_2))
        print('leng',l_1,l_2,w_1,w_2)
        rect_1 = (cx_1,cy_1), (l_1,w_1), theta_1
        rect_2 = (cx_2,cy_2), (l_2,w_2), theta_2
        box_1 = cv2.boxPoints(rect_1)
        box_2 = cv2.boxPoints(rect_2)
        angle = abs(theta_1 - theta_2)
        length_ratio = l_1 / l_2
        width_ratio =  w_1 / w_2
        dis_c = np.sqrt((cx_1-cx_2)**2+(cy_1-cy_2)**2)
        dis_m = (l_1+l_2)/2
        r = dis_c / dis_m
        dis_dif = abs(l_1-l_2)/dis_m
        y_dif = abs(cy_1-cy_2)
        x_dif = abs(cx_1-cx_2)
        y_ratio = y_dif / dis_m
        x_ratio = x_dif / dis_m
        ratio = dis_c / dis_m
        if width_ratio <= 1.5 and r >= 1 and r <= 2.7 and width_ratio >= 0.5 and dis_dif <= 0.6 and dis_dif > 0.15 and x_ratio >= 1 and y_ratio <= 1.2 and ratio >= 1 and ratio <= 4 or dis_dif == 0 and x_dif != 0:
            box_1 = cv2.boxPoints(rect_1)
            box_2 = cv2.boxPoints(rect_2)
            box_1 = box_1.reshape((-1,1,2)).astype(np.int32)
         #   cv2.polylines(img,[box_1],True,(0,255,0),5)
         #   box_2 = box_2.reshape((-1,1,2)).astype(np.int32)
         #   cv2.polylines(img,[box_2],True,(0,255,0),5)
         #   img_show('omg',img)
            print('YESSSSSSSS')
            if theta_1 > 45:
                theta_1 = 90 - theta_1
            if theta_2 >45:
                theta_2 = 90 - theta_2
            x = (cx_1+cx_2)/2
            y = (cy_1+cy_2)/2
            rotation = (theta_1+theta_2) / 2
            x1 = x - 0.5 * dis_c
            y1 = y - 0.5 * dis_m
 
            x0 = x + 0.5 * dis_c
            y0 = y1
 
            x2 = x1
            y2 = y + 0.5 * dis_m
 
            x3 = x0
            y3 = y2

            cosA = math.cos(rotation)
            sinA = math.sin(rotation)
            x0n = (x0 - x) * cosA - (y0 - y) * sinA + x
            y0n = (x0 - x) * sinA + (y0 - y) * cosA + y
 
            x1n = (x1 - x) * cosA - (y1 - y) * sinA + x
            y1n = (x1 - x) * sinA + (y1 - y) * cosA + y
 
            x2n = (x2 - x) * cosA - (y2 - y) * sinA + x
            y2n = (x2 - x) * sinA + (y2 - y) * cosA + y
 
            x3n = (x3 - x) * cosA - (y3 - y) * sinA + x
            y3n = (x3 - x) * sinA + (y3 - y) * cosA + y
 
    # 根据得到的点，画出矩形框
            cv2.line(img, (int(x0n), int(y0n)), (int(x1n), int(y1n)), (0, 255, 0), 1, 4)
            cv2.line(img, (int(x1n), int(y1n)), (int(x2n), int(y2n)), (0, 255, 0), 1, 4)
            cv2.line(img, (int(x2n), int(y2n)), (int(x3n), int(y3n)), (0, 255, 0), 1, 4)
            cv2.line(img, (int(x0n), int(y0n)), (int(x3n), int(y3n)), (0, 255, 0), 1, 4)
            point1 = [int(x0n),int(y0n)]
            point2 = [int(x1n),int(y1n)]
            point3 = [int(x2n),int(y2n)]
            pts1 = np.float32([point1,point2,point3])
            pts2 = np.float32([[500,350],[200,350],[200,500]])
            M = cv2.getAffineTransform(pts1,pts2)
            result = cv2.warpAffine(img, M,(800,800))
            img_show('1',result)
        else:
            print('none')

import cv2
import numpy as np
#import tools.infer.utility as util
#import tools.infer.predict_system as pd
np.set_printoptions(threshold=np.inf)
#args = util.parse_args(2)
#text_sys = pd.TextSystem(args)

def find_biaoqian(img,imgcopy,text_sys):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = list(contours)
    contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)

    one = np.ones_like(img) * 255
    out = np.zeros_like(img)
    out[:, :, 0] = binary
    out[:, :, 1] = binary
    out[:, :, 2] = binary

    one = cv2.drawContours(one, contours, 0, [0, 0, 0], cv2.FILLED)

    gray2 = cv2.cvtColor(one, cv2.COLOR_BGR2GRAY)
    ret2, binary2 = cv2.threshold(gray2, 128, 255, cv2.THRESH_BINARY)
    # cv2.imshow("binary", binary2)

    contours2, hierarchy2 = cv2.findContours(binary2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours3 = []
    for contour in contours2:
        if cv2.contourArea(contour) > img.shape[0] * img.shape[1] * 0.001 and cv2.contourArea(contour) < img.shape[0] * \
            img.shape[1] * 0.75:
            contours3.append(contour)

    xmin = 99999
    ymin = 99999
    xmax = 0
    ymax = 0
    for contour in contours3:
        for [[x, y]] in contour:
            if x > xmax:
                xmax = x
            if y > ymax:
                ymax = y
            if x < xmin:
                xmin = x
            if y < ymin:
                ymin = y
    length = xmax - xmin - 8
    imgcut = binary[ymin:ymax, xmin:xmax]  # h3 x5 x3

    cut3 = cv2.cvtColor(imgcut, cv2.COLOR_GRAY2RGB)
    #result = []
    #dt_boxes, rec_res = text_sys(cut3)
    #txt = ''
    #for i in range(len(rec_res)):
        #result.append([rec_res[i][0], dt_boxes[i]])
        #for j in rec_res[0][0]:
            #if j in '0123456789':
                #txt = txt + j
    txt = kedu
    print('刻度尺单位为:' + str(txt) + 'μm')

    cv2.line(imgcopy, (xmin + 6, ymin + 3), (xmax - 5, ymin + 3), [255, 0, 0], 2)
    cv2.rectangle(imgcopy, (xmin, ymin), (xmax, ymax), [0, 0, 255], 2)
    print('刻度尺长度为:' + str(length))


    return int(length),int(txt),imgcopy

def find_point(img):
    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret2, binary2 = cv2.threshold(gray2, 128, 255, cv2.THRESH_BINARY)
    # cv2.imshow("binary", binary2)

    contours2, hierarchy2 = cv2.findContours(binary2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contoursmax = []
    areamax = 0
    for contour in contours2:
        if cv2.contourArea(contour) > 0 and cv2.contourArea(contour) < img.shape[0] * \
            img.shape[1] * 0.75:
            if cv2.contourArea(contour) > areamax:
                contoursmax = contour
                areamax = cv2.contourArea(contour)
    xmin = 99999
    ymin = 99999
    xmax = 0
    ymax = 0

    for [[x, y]] in contoursmax:
        if x > xmax:
            xmax = x
        if y > ymax:
            ymax = y
        if x < xmin:
            xmin = x
        if y < ymin:
            ymin = y

    return int((xmin+xmax)/2),int((ymin+ymax)/2)

def find_point2(img):
    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret2, binary2 = cv2.threshold(gray2, 128, 255, cv2.THRESH_BINARY)
    # cv2.imshow("binary", binary2)

    contours2, hierarchy2 = cv2.findContours(binary2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contoursmax = []
    areamax = 0
    for contour in contours2:
        if cv2.contourArea(contour) > 0 and cv2.contourArea(contour) < img.shape[0] * \
            img.shape[1] * 0.75:
            if cv2.contourArea(contour) > areamax:
                contoursmax = contour
                areamax = cv2.contourArea(contour)
    xmin = 99999
    ymin = 99999
    xmax = 0
    ymax = 0

    for [[x, y]] in contoursmax:
        if x > xmax:
            xmax = x
        if y > ymax:
            ymax = y
        if x < xmin:
            xmin = x
        if y < ymin:
            ymin = y

    return xmin,ymin,xmax,ymax

def find_zuobiao(imo,changdu,danwei,imgcopy):
    imocopy = imo.copy()
    imr = np.ones(imo.shape, np.uint8)
    imp = np.ones(imo.shape, np.uint8)

    for x in range(imo.shape[0]):
        for y in range(imo.shape[1]):
            if imo[x, y][0] == 87 and imo[x, y][1] == 255 and imo[x, y][2] == 90:
                imr[x, y] = [0, 0, 255]
            if imo[x, y][0] == 255 and imo[x, y][1] == 0 and imo[x, y][2] == 255:
                imp[x, y] = [0, 255, 255]

    xr, yr = find_point(imr)
    xmin, ymin,xmax,ymax = find_point2(imr)
    xp, yp = find_point(imp)
    txt1 = '(' + str(round((xmin - xp)/changdu*danwei,1)) + ',' + str(round((ymin - yp)/changdu*danwei,1)) + ')'
    print('1:' + txt1)
    cv2.putText(imgcopy, txt1 , (int(xmin-100), int(ymin-15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),1)
    txt2 = '(' + str(round((xmax - xp) / changdu * danwei,1)) + ',' + str(round((ymin - yp) / changdu * danwei,1)) + ')'
    print('2:' + txt2)
    cv2.putText(imgcopy, txt2, (int(xmax+20), int(ymin-15)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1)
    txt3 = '(' + str(round((xmin - xp) / changdu * danwei,1)) + ',' + str(round((ymax - yp) / changdu * danwei,1)) + ')'
    print('3:' + txt3)
    cv2.putText(imgcopy, txt3, (int(xmin-100), int(ymax+15)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1)
    txt4 = '(' + str(round((xmax - xp) / changdu * danwei,1)) + ',' + str(round((ymax - yp) / changdu * danwei,1)) + ')'
    print('4:' + txt4)
    cv2.putText(imgcopy, txt4, (int(xmax+20), int(ymax+15)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1)

    cv2.circle(imgcopy, (xr, yr), 2, [0, 255, 255], -1)
    cv2.circle(imgcopy, (xp, yp), 2, [0, 255, 255], -1)
    return (xr - xp)/changdu*danwei,(yr - yp)/changdu*danwei,imgcopy,xp,yp

def draw_cankao(imd,x,y,dis20):
    start_l_x = x
    start_l_y = imd.shape[0] - 2

    start_h_x = 2
    start_h_y = y

    for i in range(-150, 150):
        x1 = int(start_l_x + i * dis20)
        y1 = int(start_l_y)
        x2 = int(start_l_x + (i + 1) * dis20)
        y2 = int(start_l_y)
        cv2.line(imd, (x1, y1), (x2, y2), (255, 255, 255), 2)
        if i % 5 == 0:
            text = str(i * 20)
            cv2.line(imd, (x1, y1), (x1, y1 - 12), (255, 255, 255), 2)
            cv2.putText(imd, text, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            cv2.line(imd, (x1, y1), (x1, y1 - 4), (255, 255, 255), 2)

    for i in range(-150, 150):
        x1 = int(start_h_x)
        y1 = int(start_h_y + i * dis20)
        x2 = int(start_h_x)
        y2 = int(start_h_y + (i + 1) * dis20)
        cv2.line(imd, (x1, y1), (x2, y2), (255, 255, 255), 2)
        if i % 5 == 0:
            text = str(i * 20)
            cv2.line(imd, (x1, y1), (x1 + 12, y1), (255, 255, 255), 2)
            cv2.putText(imd, text, (x1 + 20, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            cv2.line(imd, (x1, y1), (x1 + 4, y1), (255, 255, 255), 2)

    return imd

img = cv2.imread('./test/1111.bmp')
kedu = 200
imgcopy = img.copy()
changdu,danwei,imgcopy = find_biaoqian(img,imgcopy,kedu)
xx,yy,imgcopy,xp,yp = find_zuobiao(img,changdu,danwei,imgcopy)
print('x = ' + str(xx) + ' μm')
print('y = ' + str(yy) + ' μm')

imgcopy = draw_cankao(imgcopy,xp,yp,changdu/danwei*20)
imgcopy = cv2.resize(imgcopy,(0,0),fx=0.75,fy=0.75)
cv2.imshow('output', imgcopy)
cv2.waitKey(0)

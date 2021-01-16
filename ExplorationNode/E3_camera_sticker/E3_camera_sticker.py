import cv2
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import dlib


def detect_face(img_bgr):
    detector_hog = dlib.get_frontal_face_detector()   #- detector 선언
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rects = detector_hog(img_rgb, 2)   #- (image, num of img pyramid)
    return rects

def detect_face_with_pre_model(filename, img, rects):
    model_path = os.getcwd()+'/'+filename
    landmark_predictor = dlib.shape_predictor(model_path)
    landmarks = []
    for dlib_rect in rects:
        points = landmark_predictor(img, dlib_rect)
        list_points = list(map(lambda p: (p.x, p.y), points.parts()))
        landmarks.append(list_points)

    return landmarks

# face land mark좌표에 사각형 그리는 함수.
def draw_face_rect(img , rects):
    for dlib_rect in rects:
        l = dlib_rect.left()
        t = dlib_rect.top()
        r = dlib_rect.right()
        b = dlib_rect.bottom()
        cv2.rectangle(img, (l,t), (r,b), (0,255,0), 2, lineType=cv2.LINE_AA)

def draw_face_landmark(img, landmarks):
    for i, landmark in enumerate(landmarks):
        for idx, point in enumerate(landmarks[i]):
            cv2.circle(img, point, 2, (0, 255, 255), -1) # yellow


def show_img(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.show()

def draw_king_sticker(img, land_mark, rects):
    sticker_path = os.getcwd()+'/images/king.png'
    #sticker_path = os.getcwd()+'/images/cat-whiskers.png'

    w = rects.width()
    h = rects.height()
    x = land_mark[30][0]
    y = land_mark[30][1] - w // 2

    img_sticker = cv2.imread(sticker_path)
    img_sticker = cv2.resize(img_sticker, (w, h))

    refined_x = x - w // 2  # left, x가 nose의 x좌표였는데 rect/2 만큼 이동
    # 이럴거면 차라리 rect의 lt좌표를 쓰는게 좋지 않나? To-do : 햐우 rect의 lt로 변경해서 해보자.
    refined_y = y - h       # top, y에서 다시 스티커의 h를 빼주면

    #refined_x, y는 스티커 이미지가 그려질 왼쪽 최상단의 x,y좌표를 구하기 위함
    img_sticker = img_sticker[-refined_y:]
    refined_y = 0
    # img_show가
    sticker_area = img[refined_y:img_sticker.shape[0],
                   refined_x:refined_x+img_sticker.shape[1]]

    # 스티커 이미지에서 사용할 부분은 0 이 아닌 색이 있는 부분을 사용합니다.
    # 따라서 np.where를 통해 img_sticker 가 0 인 부분은 sticker_area를 사용하고
    # 0이 아닌 부분을 img_sticker를 사용하시면 됩니다. img_show 에 다시 적용하겠습니다
    img[refined_y:img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] = \
        np.where(img_sticker==0,sticker_area,img_sticker).astype(np.uint8)

def draw_cat_sticker(img, land_mark, rects):
    sticker_path = os.getcwd()+'/images/cat-whiskers.png'
    w = rects.width()
    h = rects.height()
    x = land_mark[30][0]
    y = land_mark[30][1]

    img_sticker = cv2.imread(sticker_path)
    img_sticker = cv2.resize(img_sticker, (h, w))

    sticker_start_x = x - img_sticker.shape[1]//2
    sticker_start_y = y - img_sticker.shape[0]//2

    sticker_area = img[sticker_start_y:sticker_start_y+img_sticker.shape[0],
                   sticker_start_x:sticker_start_x+img_sticker.shape[1]]

    img[sticker_start_y:sticker_start_y+img_sticker.shape[0], sticker_start_x:sticker_start_x+img_sticker.shape[1]] = \
        np.where(img_sticker==0, img_sticker, sticker_area).astype(np.uint8)




#my_image_path = os.getcwd()+'/images/yujin.jpg'
my_image_path = os.getcwd()+'/images/me.jpeg'
# imread의 return 좌표형태가 (y, x, c) 이렇다.
img_bgr = cv2.imread(my_image_path)    #- OpenCV로 이미지를 읽어서
# img_bgr = cv2.resize(img_bgr, (640, 360))    # 640x360의 크기로 Resize
# resize하는 경우 dlib에서 hog_detector가 제대로 동작 하지 않은 문제 발생했다.
img_show = img_bgr.copy()      #- 출력용 이미지 별도 보관

dlib_rect = detect_face(img_bgr)
#draw_face_rect(img_show, dlib_rect)
list_landmarks = detect_face_with_pre_model('shape_predictor_68_face_landmarks.dat',
                                            img_show, dlib_rect)
#draw_face_landmark(img_show, list_landmarks)
draw_cat_sticker(img_show, list_landmarks[0], dlib_rect[0])
show_img(img_show)

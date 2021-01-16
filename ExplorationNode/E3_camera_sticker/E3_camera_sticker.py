import cv2
import matplotlib.pyplot as plt
import numpy as np
import os, sys
my_image_path = os.getcwd()+'/E3_camera_sticker/images/yujin.jpg'
print(my_image_path)
img_bgr = cv2.imread(my_image_path)    #- OpenCV로 이미지를 읽어서
img_bgr = cv2.resize(img_bgr, (640, 360))    # 640x360의 크기로 Resize
img_show = img_bgr.copy()      #- 출력용 이미지 별도 보관
plt.imshow(img_bgr)
plt.show()

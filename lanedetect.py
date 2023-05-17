"""정확도 올린 차선인식 코드"""
# -*- coding: utf-8 -*-
import cv2
from matplotlib.patches import Polygon
import numpy as np #import numpy
import matplotlib.pyplot as plt #ROI 지정
import os


#cap = cv2.VideoCapture("test.mp4")
#cap = cv2.VideoCapture("drivingVideo.mov")
#cap = cv2.VideoCapture("tur1.mov")
#cap = cv2.VideoCapture("pink_steep.mov")
cap = cv2.VideoCapture("hard_test.mov")
#cap = cv2.VideoCapture("tur2.mov")
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(width, height)

def pre_processing(image):
    # Convert to HSV color space
    # hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # # Define lower and upper yellow color thresholds
    # lower_yellow = np.array([20, 100, 100])
    # upper_yellow = np.array([30, 255, 255])
    # # Mask yellow color
    # mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    hsl = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    lower_yellow = np.array([10, 0, 100])
    upper_yellow = np.array([40, 255, 255])
    # Mask yellow color
    mask_yellow = cv2.inRange(hsl, lower_yellow, upper_yellow)
    # # Mask white color
    # lower_white = np.array([0, 0, 200])
    # upper_white = np.array([255, 30, 255])
    # mask_white = cv2.inRange(hsv, lower_white, upper_white)
    # Convert masked image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Combine yellow and white masks
    mask = cv2.bitwise_or(mask_yellow, gray)
    # Apply mask to original image
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    # # Convert masked image to grayscale
    # gray = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply Canny edge detection
    edges = cv2.Canny(blur, 20, 80)
    return edges


def roi_set(image):
    x0 = int(width * 0.4)
    x01 = int(width * 0.6)
    x1 = int(width * 0.3)  # ROI의 왼쪽 하단 x좌표
    y1 = int(height * 0.6)  # ROI의 상단 y좌표
    x2 = int(width * 0.7)  # ROI의 오른쪽 x좌표
    y2 = int(height * 0.9)  # ROI의 하단 y좌표
    roi = np.array([[(x1, y1), (x2, y1), (x2, y2), (x1, y2)]])  # 좌상,우상,우하,좌ㅎㅏ
    zeros = np.zeros_like(image)
    cv2.fillPoly(zeros,roi,255) #roi 영역 흰색으로 마스킹
    roi_set = cv2.bitwise_and(image, zeros) #bitwise 연산

    return roi_set


def make_coordinates(image, xy_value): #좌표 지정
    lane_gradient, y_intecept = xy_value
    y1 = int(height) #height start from bottom
    y2 = int(y1*0.7) #높이 대비 라인이 끝나는 점 위치
    x1 = int((y1 - y_intecept)/lane_gradient) #좌표 지정
    x2 = int((y2 - y_intecept)/lane_gradient)
    return np.array([x1, y1, x2, y2])

def draw_line(image, hough_lines): #검정 배경 만들고 위에 차선 그음
    line_layer = np.zeros_like(image) #빈 어레이 만듦
    if hough_lines is not None: #라인 값이 들어올때
        for x1, y1, x2, y2 in hough_lines:
            cv2.line(line_layer, (x1, y1), (x2, y2), (0,255,0), 5) #검은 화면에 좌표1, 좌표2, 선색깔, 선두께로 라인 그음
    return line_layer

def lane_define(image, hough_lines):
    global glane_gradient
    left_raw = []
    right_raw = [] #왼/오른쪽 차선 값 임시로 받아오는 어레이
    for lane in hough_lines:
        x1, y1, x2, y2 = lane.reshape(4) #2차원 배열을 1차원으로 변환
        xy_value = np.polyfit((x1,x2),(y1,y2), 1 )
        #print(xy_value)#print [lane_gradient,  Y intecept]
        lane_gradient = xy_value[0] #파라미터 0에 기울기 저장
        y_intecept = xy_value[1] #파라미터 1에 인터셉트 값 저장
        #print(format(lane_gradient, 'f'), "lane_gradient")
        #print("----")
        #print(y_intecept, "intercep")
        glane_gradient = lane_gradient
        # if lane_gradient < 0: # 기울기 기준으로 좌/우 라인 구분
        #     left_raw.append((lane_gradient, y_intecept))
        # else:
        #     right_raw.append((lane_gradient, y_intecept))
        if -1.5 <= lane_gradient <= -0.5: # 기울기 범위 설정
            left_raw.append((lane_gradient, y_intecept))
        elif 0.5 <= lane_gradient <= 1.5:
            right_raw.append((lane_gradient, y_intecept))
    print(left_raw, 'left') #왼쪽 값
    print(right_raw, 'right') #오른쪽 값
    left_average = np.average(left_raw, axis=0)
    right_average = np.average(right_raw, axis=0)
    #print(left_average, 'left lane_gradient') #기울기 평균값
    #print(right_average, 'right lane_gradient')
    left_line = make_coordinates(image, left_average)
    right_line = make_coordinates(image, right_average)
    return np.array([left_line, right_line])


delay = 0
while(cap.isOpened()):
    _, frame = cap.read() #프레임 불러오기
    pre_processed_img = pre_processing(frame) #전처리
    cv2.imshow("pre_processed_img",pre_processed_img)
    roi_inage = roi_set(pre_processed_img) #ROI 설정
    cv2.imshow("ROI", roi_inage)
    # cv2.imshow("RAW", frame)
    hough_lines = cv2.HoughLinesP(roi_inage, 1 , np.pi/180, 20, np.array([]), minLineLength=40, maxLineGap=5)
    try:
        defined_lines = lane_define(roi_inage, hough_lines) #라인 도출
    except:
        pass
    #print(delay)
    try:
        print(abs(glane_gradient))
        print(delay)
        if not -1 <glane_gradient<0.6:
            try:
                line_output= draw_line(frame, defined_lines) #라인 시각화
                delay = 0
            except:
                delay += 1
                pass
        else:
            delay += 1
    except:
        pass
    if delay == 150:
        line_output = np.zeros_like(frame)
        delay = 0
    try:
        final_image = cv2.addWeighted(frame, 0.8, line_output, 1, 1) #이미지 두개 합침 / 이미지1, 투명도, 이미ㅈ2, 투명도, rgb 픽셀 밝기
    except:
        pass
    try:
        cv2.namedWindow("Lane_detection", flags=cv2.WINDOW_NORMAL)
        cv2.imshow("Lane_detection", final_image)
    except:
        pass
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()



"""
**정확도 올린 차선인식 코드**
영상마다 시야각이 달라서 roi조정이 필요하고 환경마다 hough threshold를 조절해주어야함!
+)pink_steep.mov의 경우 곡선차선이 많은데 곡선주행시 계속 미분을 수행해서 부드러운 곡선 차선을 만들 수 있게 하여 흔들림 개선해야함
-> 기존 코드 대비 개선점 : ROI를 다양한 비디오에도 적용 가능하도록 좀더 범용적으로 바꿨고, 색공간을 GRAYSCALE하나만 사용했던 기존 코드와 달리
흰차선은 GRAYSCALE로 한번 검출하고 노란색이나 기타 조명변화에도 감지 가능하도록 HSL색공간에서의 노란 차선을 추출하여 bitwise_or연산으로 합침
다양한 조명변화에도 대응가능 but. 푸른색이나 붉은색이 쬐~꼼 섞이면 검출이 잠깐 튀기도 함 (1~2초정도) 따라서 칼만필터같은거로 이전 프레임에서의
차선의 위치값을 참조해 차선 추적을 하도록 하든가 해야할듯
+) 차선이 띄엄띄엄있거나 사이드에 차들 때문에 차선이 가려지는 등의 문제 발생시 차선보다 앞에 보이는 노면 표시를 직선으로 검출하기도 함 (ex.화살표)
따라서 이를 해결하기 위해 간격조건을 추가하거나 양쪽 차선의 연장선을 그어서 두 직선사이의 각도에 제한을 두는 식으로 보완할 수 있을 것 같다.
"""

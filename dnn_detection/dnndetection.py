# 필요한 패키지 import
import argparse # 명령행 파싱(인자를 입력 받고 파싱, 예외처리 등) 모듈
import imutils # 파이썬 OpenCV가 제공하는 기능 중 복잡하고 사용성이 떨어지는 부분을 보완(이미지 또는 비디오 스트림 파일 처리 등)
import numpy as np # 파이썬 행렬 수식 및 수치 계산 처리 모듈
import cv2 # opencv 모듈


# 얼굴 인식 모델 로드
print("[얼굴 인식 모델 로딩]")
face_detector = "./face_detector/"
prototxt = face_detector + "deploy.prototxt.txt" # prototxt 파일 : 모델의 레이어 구성 및 속성 정의
weights = face_detector + "res10_300x300_ssd_iter_140000.caffemodel" # caffemodel 파일 : 얼굴 인식을 위해 ResNet 기본 네트워크를 사용하는 SSD(Single Shot Detector) 프레임워크를 통해 사전 훈련된 모델 가중치 사용
net = cv2.dnn.readNet(prototxt, weights) # cv2.dnn.readNet() : 네트워크를 메모리에 로드

# input 이미지 읽기
image = cv2.imread("./input/3.jpg")

# 이미지 resize
#image = imutils.resize(image, width=500)

# 이미지 크기
(H, W) = image.shape[:2]

# blob 이미지 생성
# 파라미터
# 1) image : 사용할 이미지
# 2) scalefactor : 이미지 크기 비율 지정
# 3) size : Convolutional Neural Network에서 사용할 이미지 크기를 지정
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

# 얼굴 인식
print("[얼굴 인식]")
net.setInput(blob) # setInput() : blob 이미지를 네트워크의 입력으로 설정
detections = net.forward() # forward() : 네트워크 실행(얼굴 인식)

# 인식할 최소 확률
minimum_confidence = 0.5

# 얼굴 번호
number = 0

# 얼굴 인식을 위한 반복
for i in range(0, detections.shape[2]):
    # 얼굴 인식 확률 추출
    confidence = detections[0, 0, i, 2]
    
    # 얼굴 인식 확률이 최소 확률보다 큰 경우
    if confidence > minimum_confidence:
        # bounding box 위치 계산
        box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
        (startX, startY, endX, endY) = box.astype("int")
        
        # bounding box 가 전체 좌표 내에 있는지 확인
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(W - 1, endX), min(H - 1, endY))        
        
        cv2.putText(image, "Face[{}]".format(number + 1), (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # 얼굴 번호 출력
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2) # bounding box 출력

        number = number + 1 # 얼굴 번호 증가


# 이미지 show
cv2.imshow("Face Detection", image)
# 이미지 저장
cv2.imwrite("./output/output.jpg", image) # 파일로 저장, 포맷은 확장자에 따름
cv2.waitKey(0)
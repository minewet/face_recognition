import cv2

# load model
model_path = 'models/opencv_face_detector_uint8.pb'
config_path = 'models/opencv_face_detector.pbtxt'
net = cv2.dnn.readNetFromTensorflow(model_path, config_path)

conf_threshold = 0.7

# initialize video source, default 0 (webcam)
img = cv2.imread('img/3.jpg', cv2.IMREAD_COLOR)

# prepare input
result_img = img.copy()
h, w, _ = result_img.shape
blob = cv2.dnn.blobFromImage(result_img, 1.0, (300, 300), [104, 117, 123], False, False)
net.setInput(blob)

# inference, find faces
detections = net.forward()

# postprocessing
for i in range(detections.shape[2]):
 confidence = detections[0, 0, i, 2]
 if confidence > conf_threshold:
     x1 = int(detections[0, 0, i, 3] * w)
     y1 = int(detections[0, 0, i, 4] * h)
     x2 = int(detections[0, 0, i, 5] * w)
     y2 = int(detections[0, 0, i, 6] * h)
     print(x1,y1,x2,y2)

    # draw rects
     cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), int(round(h/150)), cv2.LINE_AA)
     cv2.putText(result_img, '%.2f%%' % (confidence * 100.), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


  # visualize
cv2.imshow('result', result_img)
cv2.waitKey()
cv2.destroyAllWindows()
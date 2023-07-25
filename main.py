from ultralytics import YOLO
import cv2

model = YOLO("./models/best.pt")

def detect(frame) :
    return model(frame)[0].plot()

cap = cv2.VideoCapture(0)

while True :
    ret, frame  =cap.read()
    if not ret :
        break
    result = detect(frame)
    cv2.imshow('Object Count',result)
    if cv2.waitKey(25) & 0xFF == ord("q") :
        cv2.destroyAllWindows()
        break



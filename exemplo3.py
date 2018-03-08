import cv2

#video = cv2.VideoCapture('videos\\video.mp4')
video = cv2.VideoCapture(0)
classificador_face = cv2.CascadeClassifier('cascades\\haarcascade_frontalface_default.xml')

while True:
    conectado, frame = video.read()

    frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detectadas = classificador_face.detectMultiScale(frame_cinza)
    for (x, y, l, a) in faces_detectadas:
        cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

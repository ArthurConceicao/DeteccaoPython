import cv2

#video = cv2.VideoCapture('videos\\video.mp4')
video = cv2.VideoCapture(0)
classificador_face = cv2.CascadeClassifier('cascades\\haarcascade_frontalface_default.xml')
classificador_olhos = cv2.CascadeClassifier('cascades\\haarcascade_eye.xml')

while True:
    conectado, frame = video.read()

    frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detectadas = classificador_face.detectMultiScale(frame_cinza)
    for (x, y, l, a) in faces_detectadas:
        # imagem = cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
        regiao = frame[y:y + a, x:x + l]
        regiao_cinza_olho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
        olhos_detectados = classificador_olhos.detectMultiScale(regiao_cinza_olho, scaleFactor=1.1, minNeighbors=10)
        print(olhos_detectados)
        for (x2, y2, l2, a2) in olhos_detectados:
            frame = cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)
            regiao = cv2.rectangle(regiao, (x2, y2), (x2 + l2, y2 + a2), (255, 0, 0), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

import cv2

#classificador = cv2.CascadeClassifier('cascades\\haarcascade_frontalcatface.xml')
classificador = cv2.CascadeClassifier('cascades\\relogios.xml')

imagem = cv2.imread('outros\\relogio3.jpg')

imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

#, scaleFactor=1.02,minSize=(50,50) ,minNeighbors=1, maxSize=(80,80)
detectado = classificador.detectMultiScale(imagem_cinza)

for (x, y, l, a) in detectado:
    cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)

cv2.imshow("Detectado",imagem)
cv2.waitKey()
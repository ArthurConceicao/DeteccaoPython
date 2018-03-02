import cv2

classificador = cv2.CascadeClassifier('C:\Deteccao\cascades\haarcascade_frontalface_default.xml')

imagem = cv2.imread('pessoas\\pessoas3.jpg')
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

faces_detectadas = classificador.detectMultiScale(imagem_cinza, scaleFactor=1.1, minNeighbors=9, minSize=(30, 30))
print(len(faces_detectadas))
print(faces_detectadas)

for(x, y, l, a) in faces_detectadas:
    print(x, y, l, a)
    cv2.rectangle(imagem, (x, y), (x+l, y+a), (0, 0, 255), 2)

cv2.imshow("Faces encontradas", imagem)
cv2.waitKey()
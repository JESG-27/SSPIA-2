import cv2 as cv
import tensorflow as tf
from tensorflow import keras
import numpy as np

cascPath = "datasets/haarcascade_frontalface_default.xml"
faceCascade = cv.CascadeClassifier(cascPath)
model = keras.models.load_model("smileornot.h5")

videoCapture = cv.VideoCapture(0, cv.CAP_DSHOW)
anterior = 0
if not videoCapture.isOpened():
    print("No se puede abrir la cámara")
else:
    while True:
        ret, frame = videoCapture.read()
        frame = cv.flip(frame, 1)
        imagenGris = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            imagenGris, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(50, 50), 
            flags=cv.CASCADE_SCALE_IMAGE)

        imagenRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        for (x, y, h, w) in faces:
            rostro = cv.resize(frame[y:y+h, x:x+w], (64, 64))
            cv.imshow("rostro", rostro)
            rostroRGB = cv.resize(imagenRGB[y:y+h, x:x+w, :], (64, 64))
            rostroRGB = rostroRGB / 255.0
            classes = ["smile", "non_smile"]
            output = model.predict(np.array([rostroRGB]))
            print(output)
            output = np.argmax(output, axis=1)
            print(output)
            print(classes[output.squeeze()])
            cv.rectangle(frame, (x, y-5), (x+w, y+h), (0, 255, 0), 1)
            cv.putText(frame, classes[output.squeeze()], (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.imshow("Video", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    videoCapture.release()
    cv.destroyAllWindows()

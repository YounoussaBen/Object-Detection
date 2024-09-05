import streamlit as st
import cv2
import numpy as np
import tempfile

#Opencv DNN
net = cv2.dnn.readNet("dashboard/model/dnn_model/yolov4-tiny.weights", "dashboard/model/dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size =(320, 320), scale=1/255)

#cargar class list
classes = []
with open("dashboard/model/dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

st.markdown("<h1 style='text-decoration: underline;'>Sidewalk Obstacle Detection and Navigation Assistance for the Visually Impaired</h1>", unsafe_allow_html=True)



#video_file = st.sidebar.file_uploader("", type=["mp4", "avi", "mov"])

# Guardar el archivo en un directorio temporal
#temp_video = tempfile.NamedTemporaryFile(delete=False)
#temp_video.write(video_file.read())
#temp_video_path = temp_video.name

st.title("Obstacle Detection")
st.write("press and activate your camera to start")

if st.button("Obstacle Detection"):
    #iniciar camara
    cap = cv2.VideoCapture(0)
    #Parametros de la camara
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


    FRAME_WINDOW = st.image([])    

    while True:
      ret, frame = cap.read()

      # convertir la imagen a RGB
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

      

      #object detection
      
      (class_ids, score, bboxes) = model.detect(frame)


      if len(class_ids) == 0:
        print("No se detectaron objetos en este frame.")

      for class_id, score, bbox in zip(class_ids, score, bboxes):
        (x,y,w,h) = bbox
      class_name = classes[class_id]

      #Colocar el nombre de las clases
      cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (200, 0, 50), 2)
      #colocar el rectangulo
      cv2.rectangle(frame, (x,y), (x+w, y+h), (200, 0, 50), 3)

      if frame.shape[0] > 0 and frame.shape[1] > 0:
        FRAME_WINDOW.image(frame)
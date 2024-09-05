import cv2
import streamlit as st
import numpy as np
import torch

# Cargar el modelo YOLOv7
model = torch.load('model/YOLOV7-main/yolov7.pt', map_location='cpu')

# Función para la detección de objetos
def detect_objects(image):
    # Convertir la imagen a un formato compatible con el modelo YOLOv7
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0).unsqueeze(0)

    # Ejecutar el modelo YOLOv7 en la imagen
    outputs = model(image_tensor)

    # Obtener las predicciones y convertirlas a coordenadas de cajas de detección
    predictions = outputs.xyxy[0].cpu().numpy()

    # Regresar las predicciones
    return predictions

# Función para mostrar la imagen y las predicciones en la aplicación Streamlit
def show_image_with_predictions(image, predictions):
    # Dibujar las cajas de detección en la imagen
    for pred in predictions:
        xmin, ymin, xmax, ymax, conf, class_id = pred
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, f"{class_id} {conf:.2f}", (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar la imagen en la aplicación Streamlit
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="BGR")

# Configurar la aplicación Streamlit
st.title("Clasificación de objetos en vivo con YOLOv7")
video = st.video("https://www.sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4")

# Capturar el video en vivo
cap = cv2.VideoCapture(0)

# Procesar los frames del video en vivo
while True:
    # Capturar un frame del video
    ret, frame = cap.read()

    # Detectar objetos en el frame
    predictions = detect_objects(frame)

    # Mostrar el frame con las predicciones
    show_image_with_predictions(frame, predictions)

    # Salir del loop si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y destruir las ventanas de OpenCV
cap.release()
cv2.destroyAllWindows()
import streamlit as st
import cv2
import numpy as np
import tempfile

# Opencv DNN
net = cv2.dnn.readNet("dashboard/model/dnn_model/yolov4-tiny.weights", "dashboard/model/dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

# Load class list
classes = []
with open("dashboard/model/dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

# Set page config for mobile-friendly layout
st.set_page_config(layout="wide")

# Use columns for better mobile layout
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("<h1 style='font-size: 24px; text-align: center;'>Sidewalk Obstacle Detection</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='font-size: 18px; text-align: center;'>Navigation Assistance for the Visually Impaired</h2>", unsafe_allow_html=True)

    st.write("Press the button to activate your camera and start detection")

    # Create a placeholder for the video feed
    video_placeholder = st.empty()

with col2:
    if st.button("Start Detection", use_container_width=True):
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video. Please check your camera.")
                break

            # Convert the image to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Object detection
            (class_ids, scores, bboxes) = model.detect(frame)

            for class_id, score, bbox in zip(class_ids, scores, bboxes):
                (x, y, w, h) = bbox
                class_name = classes[class_id]

                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x+w, y+h), (200, 0, 50), 2)
                label = f"{class_name}: {score:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 50), 2)

            # Display the frame
            video_placeholder.image(frame, use_column_width=True)

    if st.button("Stop Detection", use_container_width=True):
        st.stop()

# Add some spacing
st.markdown("<br><br>", unsafe_allow_html=True)

# Instructions or additional information
st.markdown("### How to use:")
st.write("1. Press 'Start Detection' to begin obstacle detection.")
st.write("2. Ensure your camera is properly connected and accessible.")
st.write("3. Press 'Stop Detection' to end the session.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>Developed for assisting visually impaired individuals</p>", unsafe_allow_html=True)
import cv2

#Opencv DNN
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size =(320, 320), scale=1/255)

#cargar class list
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

print("objects list")
print(classes)

#iniciar camara
cap = cv2.VideoCapture(0)
#Parametros de la camara
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#FULL HD 1920 x 1080 (si es necesario)


while True:
    #conseguir frames
    ret, frame = cap.read()

    #object detection
    (class_ids, score, bboxes) = model.detect(frame)
    for class_id, score, bbox in zip(class_ids, score, bboxes):
        (x,y,w,h) = bbox
        class_name = classes[class_id]
        
        #Colocar el nombre de las clases
        cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (200, 0, 50), 2)
        #colocar el rectangulo
        cv2.rectangle(frame, (x,y), (x+w, y+h), (200, 0, 50), 3)




    print("class ids", class_ids)
    print("scores", score)
    print("bboxes", bboxes)


    
    cv2.imshow("frame", frame)
    cv2.waitKey(1)




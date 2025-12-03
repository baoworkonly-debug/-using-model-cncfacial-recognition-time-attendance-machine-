import tensorflow as tf
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

seved_model = tf.keras.models.load_model("E:\code\demo\project\lvtn\pretrained_models\COLAP\model\model.h5")

import cv2
import cham_cong

cap = cv2.VideoCapture(0)

dir_class_name = r'E:\code\demo\project\lvtn\pretrained_models\COLAP\data\class.txt'
classNames =[]
# Đọc file text dòng 
with open(dir_class_name, 'r') as file:
    for line in file:
        classNames.append(line.strip())
    
print(classNames)
file_txt = r'E:\code\demo\project\lvtn\weights-prototxt.txt'
file_model = r'E:\code\demo\project\lvtn\res_ssd_300Dim.caffeModel'
net = cv2.dnn.readNetFromCaffe(file_txt, file_model)
check_chamcong= []
anable_chamcong =0
def predict_face(frame,x1,x2,y1,y2):
    
    
    cropped = frame[y1:y2, x1:x2]
    if cropped.size > 0:
        cropped = cv2.resize(cropped, (64, 64))
    else:
        # Xử lý trường hợp cropped là rỗng
        return '??????'
        
    
    image_ = cropped.astype(np.uint8)

    image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)

    image_ = image_.astype("float") / 255.0
    image_ = np.expand_dims(image_, axis=0)

    results = seved_model.predict(image_)
    confidence = np.max(results, axis=1)

    for i, pred in enumerate(results):
        print(f"Ảnh {i+1}:")
        print(f"  Dự đoán: {np.argmax(pred)}")
        print(f"  Độ tin cậy: {confidence[i]*100:.2f}%")
    confi = confidence[i]*100
    if 72 < confi < 90:
    
        final = np.argmax(results)
        print(classNames[final])
        name_final = classNames[final]

        
        for i in check_chamcong:
            if i == final:
                return name_final
        print("------------------------------------------------")
        print("da cham cong")
        print("------------------------------------------------")
        check_chamcong.append(final)
        cham_cong.chamcong(name_final,final)
        return name_final
    else :
        return '????'
def test_predict():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        
        # chuyển ảnh xám
        (height, width) = frame.shape[:2]
    
        
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))
    
        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence < 0.5:
                continue


    
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x1, y1, x2, y2) = box.astype("int")

            #text = "{:.2f}%".format(confidence * 100) + " ( " + str(y2-y1) + ", " + str(x2-x1) + " )"
            y = y1 - 10 if y1 - 10 > 10 else y1 + 10

            cv2.rectangle(frame, (x1, y1), (x2, y2),
                (0, 0, 255), 2)
            
            #---------------------------------------------------------------------------------------------------------
            name_predict =''
            if(x1>200 & x2<455 & (y2 - y1) >250):
                print(f'{x1}:{x2}:{y1}:{y2}')
                if frame is None:
                    continue
                name_predict = predict_face(frame,x1,x2,y1,y2)
                #print(name_predict)
            if name_predict is None:   
                name_predict ="????"
            
            #text = f'{text}-name: {name_predict}'
            text_name = f'name: {name_predict}'
            cv2.putText(frame, text_name, (x1, y),
                cv2.LINE_AA, 0.45, (0, 0, 255), 2)
            
            #---------------------------------------------------------------------------------------------------------
   
        cv2.imshow('Face Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Release the camera and close all windows




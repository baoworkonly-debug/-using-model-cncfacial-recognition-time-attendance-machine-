
import numpy as np
import cv2
import os

file_txt = r'E:\code\demo\project\lvtn\weights-prototxt.txt'
file_model = r'E:\code\demo\project\lvtn\res_ssd_300Dim.caffeModel'





def cut_image_from_video(vs,name):
    stat_cut =0
    num =0
    net = cv2.dnn.readNetFromCaffe(file_txt, file_model)
    while True:
        ret, frame =    vs.read()          
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

      
            text = "{:.2f}%".format(confidence * 100) + " ( " + str(y2-y1) + ", " + str(x2-x1) + " )"
 
            y = y1 - 10 if y1 - 10 > 10 else y1 + 10
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                (0, 0, 255), 2)
            cv2.putText(frame, text, (x1, y),
                cv2.LINE_AA, 0.45, (0, 0, 255), 2)
            #---------------------------------------------------------------------------------------------------------
            cropped = frame[y1:y2, x1:x2]
            cropped = frame            
            save_path = f"E:\code\demo\project\lvtn\data\IMAGE\FACE_test\{name}"              
            if not os.path.exists(save_path):              
                os.makedirs(save_path)
                print(f"Thư mục '{save_path}' đã được tạo.")
                 
            if stat_cut == 1:
                num = num + 1
                cv2.imwrite(os.path.join(save_path, f'captured_image{num}.jpg'), cropped)
                state = f"Đã chụp ảnh và lưu vào tệp captured_image{ num}.jpg"
                print(state)
                if num % 100 ==0:
                    stat_cut =0
            

#---------------------------------------------------------------------------------------------------------
           

    
        cv2.imshow("Window", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            stat_cut =1
      
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    vs.stop()           


#video = cv2.VideoCapture(r'E:\code\demo\project\lvtn\data\video\1.mp4')
video =cv2.VideoCapture(0)
cut_image_from_video(video,"1")



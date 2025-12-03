import threading
import cv2
import os
from flask import Flask, render_template, Response, redirect, session, request
import numpy as np
app = Flask(__name__)
import predict
import requests
import cham_cong
# def cut(stat_cut,x1,x2,y1,y2,frame,name,num):
#     if stat_cut == 1 & (y2-y1) >250:
#         cropped = frame[y1:y2, x1:x2]
#         cropped = frame
#         print("------------------------")
#         print(name)
#         print("------------------------")
#         # Tạo đường dẫn đến thư mục
#         if(name != "000"):
            
#             save_path = f"E:/code/demo/project/lvtn/pretrained_models/COLAP/face/train/{name}"

#             # Kiểm tra xem thư mục có tồn tại không, nếu không thì tạo mới
#             if not os.path.exists(save_path):
#                 os.makedirs(save_path)
            
#             cv2.imwrite(os.path.join(save_path, f'captured_{name}_face{self.num}.jpg'), cropped)
#             state = f"Đã chụp ảnh và lưu vào tệp captured_image{num}.jpg"
#             print(state)
#             if num % 100 ==0:
#                 stat_cut =0
#                 name = "000"
#                 return stat_cut,name
#             else:
#                 return stat_cut,name
                
class VideoStream:
    def __init__(self):
        self.video_capture = cv2.VideoCapture(0)
        self.frame = None
        self.is_running = False
        self.thread = None
        self.file_txt = r'E:\code\demo\project\lvtn\weights-prototxt.txt'
        self.file_model = r'E:\code\demo\project\lvtn\res_ssd_300Dim.caffeModel'
        self.net = cv2.dnn.readNetFromCaffe(self.file_txt, self.file_model)
        self.name = "000"
        self.stat_cut = 0
        self.num = 0
        self.classNames =[]
        self.url = 'http://192.168.1.5/480x320.jpg'
    def c(self):
        dir_class_name = r'E:\code\demo\project\lvtn\pretrained_models\COLAP\data\class.txt'
        with open(dir_class_name, 'r') as file:
            for line in file:
                self.classNames.append(line.strip())
    def clas_name(self):
    
        
        return self.classNames
    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self._update_frame)
        self.thread.start()

    def stop(self):
        self.is_running = False
        self.thread.join()

    def get_frame(self):
        return self.frame
    def cut_image(self):
        self.stat_cut =1 
    def self_name(self, name):
        self.name = name
    def _update_frame(self):
        while self.is_running:
            
            ret, frame = self.video_capture.read()
            
            if not ret:
                break
           
            
            # response = requests.get('http://192.168.1.5/480x320.jpg')
    
   
            # img_arr = np.frombuffer(response.content, np.uint8)
            
            # # Giải mã ảnh thành frame
            # frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)


            (height, width) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))

        # pass the blob into dnn 
            self.net.setInput(blob)
            detections = self.net.forward()
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence < 0.75:
                    continue
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (x1, y1, x2, y2) = box.astype("int")


                
                # --------------------------------------------------
                # 
                #text = "{:.2f}%".format(confidence * 100) + " ( " + str(y2-y1) + ", " + str(x2-x1) + " )"
                y = y1 - 10 if y1 - 10 > 10 else y1 + 10
                
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                    (0, 0, 255), 2)
                name_predict =''
                if(x1>200 & x2<455 & (y2 - y1) >250):
                    print(f'{x1}:{x2}:{y1}:{y2}')
                    if frame is None:
                        continue
                    name_predict = predict.predict_face(frame,x1,x2,y1,y2)
                    #print(name_predict)
                if name_predict is None:   
                    name_predict ="????"
                
                #text = f'{text}-name: {name_predict}'
                text_name = f'name: {name_predict}'
                cv2.putText(frame, text_name, (x1, y),
                    cv2.LINE_AA, 0.45, (0, 0, 255), 2)
            
                #---------------------------------------------------------------------------------------------------------
                

                #Lưu ảnh vào t;hư mục
                
                        
            #cv2.imshow("Window", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('p'):
                self.stat_cut =1

            if key == ord('q'):
                break

            self.frame = frame



@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = video_stream.get_frame()
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    url = r"E:\code\demo\project\lvtn\pretrained_models\COLAP\data\train"
    folders = []
    for label_dir in os.listdir(url):
        label_path = os.path.join(url, label_dir)
        if os.path.isdir(label_path):
            folders.append(label_dir)
            print("label_dir")


    return render_template('index.html', folders=folders)
@app.route('/process_data', methods=['POST'])
def process_data():
 #-------------------------- --------------------------------------------- 
    folders = video_stream.clas_name()
    
   
 #-------------------------- ---------------------------------------------           
    input_data = request.form['input_data']
    
    processed_data = input_data.upper()
    print(f"Dữ liệu được nhập: {input_data}")
    print(f"Dữ liệu đã được xử lý: {processed_data}")
    
    video_stream.self_name(input_data)
    video_stream.cut_image()
    # new_folder_path = r"E:\code\demo\project\lvtn\pretrained_models\COLAP\face\train\4"
    # os.makedirs(new_folder_path, exist_ok=True)


    
    return render_template('index.html', processed_data=processed_data,folders=folders)


# @app.route('/do_something_1', methods=['POST'])
# def do_something_1():
#     # Thực hiện công việc 1
#     result = "da lưu lại ảnh "
    
#     print(result)
#     #video_stream.cut_image()
#     return result   http://127.0.0.1:5000/submit?name=gfgfg&email=gfgfgfg

@app.route('/submit', methods=['GET'])
def handle_get():
    name = request.args.get('name')
    folders = video_stream.clas_name()
    
  
    print(f"Name: {name}")

    print(folders[int(name)])
    cham_cong.chamcong(folders[int(name)])
    return 'Data received'

if __name__ == '__main__':
    video_stream = VideoStream()
    video_stream.start()
    video_stream.c()
    app.run()
    video_stream.stop()
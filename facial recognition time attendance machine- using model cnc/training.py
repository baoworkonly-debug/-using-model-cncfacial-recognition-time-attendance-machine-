import os
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer

from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD

from tensorflow.keras.preprocessing.image import ImageDataGenerator


data_train = r"E:\code\demo\project\lvtn\pretrained_models\COLAP\face\train"  # thư mục chứa các thư mục con
data_valid = r"E:\code\demo\project\lvtn\pretrained_models\COLAP\face\validate"  # thư mục chứa các thư mục con
data_test = r"E:\code\demo\project\lvtn\pretrained_models\COLAP\face\test"  # thư mục chứa các thư mục con


import os
import numpy as np
from PIL import Image

def add_data(data_dir, w, h):
    train_images = []
    train_labels = []
    for label_dir in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_dir)
        
        # Chỉ xử lý các thư mục, bỏ qua các file
        if os.path.isdir(label_path):
            # Duyệt qua các file trong thư mục hiện tại
            for filename in os.listdir(label_path):
                image_path = os.path.join(label_path, filename)
                
                # Đọc ảnh và chuyển đổi thành numpy array
                image = Image.open(image_path)
                new_size = (w, h)  # Kích thước mới mà bạn muốn resize
                image = image.resize(new_size)
                image_array = np.array(image)
                
                # Thêm ảnh và nhãn tương ứng vào numpy array
                train_images.append(image_array)
                train_labels.append(label_dir)
    
    # Chuyển train_images và train_labels thành numpy array
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    
    return train_images, train_labels

trainx, trainy = add_data(data_train, 64, 64)
validx, validy = add_data(data_valid, 64, 64)
testx, testy = add_data(data_test, 64, 64)

# Trộn dữ liệu huấn luyện
trainx, trainy = shuffle(trainx, trainy)

# Chuyển đổi dữ liệu thành float và chia cho 255 (chuẩn hóa)
trainx = trainx.astype("float") / 255.0
validx = validx.astype("float") / 255.0
testx = testx.astype("float") / 255.0
print(trainy)


lb = LabelBinarizer()
trainyt = lb.fit_transform(trainy)
validyt = lb.fit_transform(validy)
testyt = lb.fit_transform(testy)

width = 64
height = 64



model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(3, activation='sigmoid'))
model.summary()


aug = ImageDataGenerator(
	rotation_range=0.18,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	horizontal_flip=True,)

epochs = 10
batch_size = 20
learning_rate = 0.01
opt = SGD(learning_rate=learning_rate, momentum=0.9)

model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

H = model.fit(
	aug.flow(trainx, trainyt, batch_size=batch_size),
	validation_data=(validx, validyt),
	steps_per_epoch=219//20,
	epochs=epochs)
model.save(r"E:\code\demo\project\lvtn\pretrained_models\COLAP\model\model.h5")
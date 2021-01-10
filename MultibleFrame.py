import tensorflow as tf



from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras import Sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tqdm import tqdm

datacsv = pd.read_csv('train.csv')
# print(datacsv.head(10))
#print(datacsv.head(10))
# print(datacsv.shape[0])
pathimage = 'Images'
width = 400
height = 400
# imagef = image.load_img(path= pathimage[], target_size=(width, height, 3))
# print(imagef.shape)
# print(imagef)
X = []

# print(datacsv['Id'][2] + '.jpg')

for i in tqdm(range(datacsv.shape[0])):
    path = 'C:/Users/Phuc/PycharmProjects/pythonProject5/Images/' + datacsv['Id'][i] + '.jpg'
    imagef = image.load_img(path, target_size=(width, height))
    imagef = image.img_to_array(imagef)
    imagef = imagef/255.0
    X.append(imagef)

X = np.array(X)

# print(A.shape)
# plt.imshow(A[1])
# plt.imsave('Phucdeptrai.jpg', A[1])
# datacsv['Genre'][1]

X.shape

plt.imshow(X[2250])

y = datacsv.drop(['Id', 'Genre'], axis = 1)
y = y.to_numpy()
y.shape

X_train , X_test , y_train , y_test =  train_test_split(X,y,random_state = 0,test_size = 0.15)
X_train[0].shape

model = Sequential()
model.add(Conv2D(16, (3,3), activation='relu', input_shape = X_train[0].shape))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Dropout(0.3))

model.add(Conv2D(32, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Dropout(0.3))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Dropout(0.4))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Dropout(0.5))

model.add(Conv2D(256, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Dropout(0.6))

model.add(Conv2D(512, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Dropout(0.7))

model.add(Conv2D(1024, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Dropout(0.8))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))


model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))


model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(25, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test))


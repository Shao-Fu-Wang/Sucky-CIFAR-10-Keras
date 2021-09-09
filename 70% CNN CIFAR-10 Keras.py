import tensorflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense,Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#%matplotlib inline

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel("Acc")
    plt.xlabel("Epoch")
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show
    

def plot_images_labels_prediction(images, labels, prediction, idx, num = 10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num>25:
        num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1+i)
        ax.imshow(images[idx], cmap = 'binary')
        title = str(i)+", " + label_dict[labels[i][0]]
        if len(prediction)>0:
            title += "=> " + label_dict[prediction[i]]
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx+=1
    plt.show()
    
#----------------------------------------------------------------------------------------------


(x_train_image, y_train_label), (x_test_image, y_test_label) = cifar10.load_data()
label_dict = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}


x_Train_normalize = x_train_image.astype('float32')/255.0
x_Test_normalize = x_test_image.astype('float32')/255.0
y_Train_OneHot = np_utils.to_categorical(y_train_label)
y_Test_OneHot = np_utils.to_categorical(y_test_label)

model = Sequential()

model.add(Conv2D(filters=32,
                 kernel_size=(3, 3),
                 padding='same',
                 input_shape=(32, 32, 3),
                 activation='tanh'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64,
                 kernel_size=(3, 3),
                 padding='same',
                 activation='tanh'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=128,
                 kernel_size=(3, 3),
                 padding='same',
                 activation='tanh'))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(3000, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1500, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

model.summary()


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
train_history = model.fit(x=x_Train_normalize,
                          y=y_Train_OneHot, 
                          validation_split=0.2,
                          epochs=50,
                          batch_size=128,
                          verbose=2)






show_train_history(train_history, 'accuracy', 'val_accuracy')

print()
print()
scores = model.evaluate(x_Test_normalize, y_Test_OneHot)
print()
print('accuracy = ', scores[1])
print('loss = ', scores[0])



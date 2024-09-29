## 2024.09.28. 16:00

Change:
- added class weight balancing and more evaluation

Stats:
- Accuracy of the model is -  97.86670207977295 % 

                precision    recall  f1-score   support

           0       0.96      1.00      0.98       331
           1       1.00      1.00      1.00       432
           2       1.00      0.99      1.00       310
           3       0.98      1.00      0.99       245
           4       0.98      0.87      0.92       498
           5       1.00      1.00      1.00       247
           6       0.99      1.00      1.00       348
           7       1.00      1.00      1.00       436
           8       1.00      1.00      1.00       288
           9       1.00      1.00      1.00       331
          10       1.00      1.00      1.00       209
          11       0.94      0.90      0.92       394
          12       0.92      1.00      0.96       291
          13       0.99      1.00      1.00       246
          14       1.00      1.00      1.00       347
          15       1.00      1.00      1.00       164
          16       0.99      0.98      0.99       144
          17       0.78      0.90      0.84       246
          18       1.00      0.97      0.98       248
          19       1.00      0.98      0.99       266
          20       0.97      1.00      0.99       346
          21       1.00      0.98      0.99       206
          22       1.00      1.00      1.00       267
          23       1.00      1.00      1.00       332
       accuracy                           0.98      7172
       macro avg       0.98      0.98      0.98      7172
       weighted avg       0.98      0.98      0.98      7172


```
import pandas as pd
import numpy as np
from keras.src.models import Sequential
from keras.src.layers import Dense, Conv2D, Flatten , Dropout , BatchNormalization
from keras.src.layers.pooling.max_pooling2d import MaxPooling2D
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

train_df = pd.read_csv("dataset/sign_mnist_train.csv")
test_df = pd.read_csv("dataset/sign_mnist_test.csv")

y_train = train_df['label']
y_test = test_df['label']
del train_df['label']
del test_df['label']

# Binarizing for multi-class classification
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.fit_transform(y_test)

# Data converting to NumPy arrays
x_train = train_df.values
x_test = test_df.values

# Pixel value normalizing
x_train = x_train / 255
x_test = x_test / 255

# Reshaping the data to fit the model input shape
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 1, verbose=1,factor=0.5, min_lr=0.00001)

# Convolution
model = Sequential()

# First convolutional layer
model.add(Conv2D(75 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (28,28,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2) , strides = 2 , padding = 'same'))

# Second convolutional layer
model.add(Conv2D(50 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2) , strides = 2 , padding = 'same'))

# Third convolutional layer
model.add(Conv2D(25 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2) , strides = 2 , padding = 'same'))

# Flattening from 3D to 1D
model.add(Flatten())

model.add(Dense(units = 512 , activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(units = 24 , activation = 'softmax'))
model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
model.summary()

# Calculateing class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train.argmax(axis=1)), y=y_train.argmax(axis=1))
class_weights = {i: class_weights[i] for i in range(len(class_weights))}

# Train to model
history = model.fit(datagen.flow(x_train,y_train, batch_size = 128), epochs = 20, validation_data = (x_test, y_test), class_weight=class_weights, callbacks = [learning_rate_reduction])

# Save the trained model
model.save('models/smnist.keras')

print("Accuracy of the model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")

y_pred = model.predict(x_test)
y_pred_classes = y_pred.argmax(axis=1)
y_true = y_test.argmax(axis=1)

print(classification_report(y_true, y_pred_classes))
```


## 2024.09.28. 20:00

Changed:
- added back forth conv layer and second dense

Stats:
- Accuracy of the model is -  99.9023973941803 % 

              precision    recall  f1-score   support

           0       1.00      1.00      1.00       331
           1       1.00      1.00      1.00       432
           2       1.00      1.00      1.00       310
           3       1.00      0.97      0.99       245
           4       1.00      1.00      1.00       498
           5       1.00      1.00      1.00       247
           6       1.00      1.00      1.00       348
           7       1.00      1.00      1.00       436
           8       1.00      1.00      1.00       288
           9       1.00      1.00      1.00       331
          10       0.97      1.00      0.99       209
          11       1.00      1.00      1.00       394
          12       1.00      1.00      1.00       291
          13       1.00      1.00      1.00       246
          14       1.00      1.00      1.00       347
          15       1.00      1.00      1.00       164
          16       0.99      1.00      1.00       144
          17       1.00      1.00      1.00       246
          18       1.00      1.00      1.00       248
          19       1.00      1.00      1.00       266
          20       1.00      1.00      1.00       346
          21       1.00      1.00      1.00       206
          22       1.00      1.00      1.00       267
          23       1.00      1.00      1.00       332

        accuracy                           1.00      7172
        macro avg       1.00      1.00      1.00      7172
        weighted avg       1.00      1.00      1.00      7172


```
import pandas as pd
import numpy as np
from keras.src.models import Sequential
from keras.src.layers import Dense, Conv2D, Flatten , Dropout , BatchNormalization
from keras.src.layers.pooling.max_pooling2d import MaxPooling2D
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

train_df = pd.read_csv("dataset/sign_mnist_train.csv")
test_df = pd.read_csv("dataset/sign_mnist_test.csv")

y_train = train_df['label']
y_test = test_df['label']
del train_df['label']
del test_df['label']

# Binarizing for multi-class classification
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.fit_transform(y_test)

# Data converting to NumPy arrays
x_train = train_df.values
x_test = test_df.values

# Pixel value normalizing
x_train = x_train / 255
x_test = x_test / 255

# Reshaping the data to fit the model input shape
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 1, verbose=1,factor=0.5, min_lr=0.00001)

# Convolution
model = Sequential()

# First convolutional layer
model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), padding='same'))

# Second convolutional layer
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), padding='same'))

# Third convolutional layer
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), padding='same'))

# Fourth convolutional layer
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), padding='same'))

model.add(Flatten())

# First dense layer
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))

# Second dense layer
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(units = 24 , activation = 'softmax'))
model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
model.summary()

# Calculateing class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train.argmax(axis=1)), y=y_train.argmax(axis=1))
class_weights = {i: class_weights[i] for i in range(len(class_weights))}

# Train to model
history = model.fit(datagen.flow(x_train,y_train, batch_size = 128), epochs = 20, validation_data = (x_test, y_test), class_weight=class_weights, callbacks = [learning_rate_reduction])

# Save the trained model
model.save('models/smnist.keras')

print("Accuracy of the model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")

y_pred = model.predict(x_test)
y_pred_classes = y_pred.argmax(axis=1)
y_true = y_test.argmax(axis=1)

print(classification_report(y_true, y_pred_classes))
```
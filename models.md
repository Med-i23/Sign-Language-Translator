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


## 2024.11.18. 19:00

### Changes
- Switch to ASL database
- Using CUDA now
- Tensorflow 2.10

### Things to note
- Train time less then 7 hours, approx. 5-6
- Early stopping at epoch 27. Best score

**Evaluation**
```
The accuracy of the model for training data is: 99.54454302787781
The Loss of the model for training data is: 0.01568509079515934
The accuracy of the model for validation data is: 99.44827556610107
The Loss of the model for validation data is: 0.018894536420702934
The accuracy of the model for testing data is: 99.47126507759094
The Loss of the model for testing data is: 0.02074027992784977

              precision    recall  f1-score   support

           A       0.98      1.00      0.99       300
           B       0.99      0.99      0.99       300
           C       1.00      1.00      1.00       300
           D       1.00      0.99      0.99       300
           E       0.98      0.99      0.99       300
           F       1.00      1.00      1.00       300
           G       0.99      0.99      0.99       300
           H       0.99      1.00      1.00       300
           I       1.00      1.00      1.00       300
           G       1.00      0.99      0.99       300
           K       0.99      1.00      1.00       300
           L       1.00      1.00      1.00       300
           M       0.98      0.98      0.98       300
           N       0.99      0.98      0.98       300
           O       0.99      1.00      1.00       300
           P       1.00      1.00      1.00       300
           Q       1.00      1.00      1.00       300
           R       1.00      0.99      0.99       300
           S       0.99      1.00      0.99       300
           T       1.00      1.00      1.00       300
           U       0.98      1.00      0.99       300
           V       0.99      0.99      0.99       300
           W       1.00      0.99      1.00       300
           X       1.00      0.98      0.99       300
           Y       0.99      1.00      1.00       300
           Z       0.99      0.99      0.99       300
         del       1.00      1.00      1.00       300
     nothing       1.00      1.00      1.00       300
       space       1.00      1.00      1.00       300

    accuracy                           0.99      8700
   macro avg       0.99      0.99      0.99      8700
weighted avg       0.99      0.99      0.99      8700
```

**Code**

```

import os
import numpy as np
import pandas as pd
import splitfolders
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix


base_path = "datasets/asl_main/asl_alphabet_train/asl_alphabet_train/"

categories = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H",
    8: "I", 9: "G", 10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P",
    16: "Q", 17: "R", 18: "S", 19: "T", 20: "U", 21: "V", 22: "W", 23: "X",
    24: "Y", 25: "Z", 26: "del", 27: "nothing", 28: "space",
}

filenames_list = []

categories_list = []

for category in categories:
    filenames = os.listdir(base_path + categories[category])
    filenames_list = filenames_list + filenames
    categories_list = categories_list + [category] * len(filenames)

df = pd.DataFrame({"filename": filenames_list, "category": categories_list})

df = df.sample(frac=1).reset_index(drop=True)

# Only run on the first setup, after that comment it out
# splitfolders.ratio('datasets/asl_main/asl_alphabet_train/asl_alphabet_train', output='workdir/', seed=1333, ratio=(0.8, 0.1, 0.1))

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='nearest'
)


train_path = 'workdir/train'
val_path = 'workdir/val'
test_path = 'workdir/test'

batch = 32
image_size = 200
img_channel = 3
n_classes = 29

train_data = datagen.flow_from_directory(directory=train_path, target_size=(image_size, image_size), batch_size=batch, class_mode='categorical')
val_data = datagen.flow_from_directory(directory=val_path, target_size=(image_size, image_size), batch_size=batch, class_mode='categorical')
test_data = datagen.flow_from_directory(directory=test_path, target_size=(image_size, image_size), batch_size=batch, class_mode='categorical', shuffle=False)

model = Sequential()
# Conv layer 1
model.add(Conv2D(32, 3, activation='relu', padding='same', input_shape=(image_size, image_size, img_channel)))
model.add(Conv2D(32, 3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(padding='same'))
model.add(Dropout(0.2))

# Conv layer 2
model.add(Conv2D(64, 3, activation='relu', padding='same'))
model.add(Conv2D(64, 3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(padding='same'))
model.add(Dropout(0.3))

# Conv layer 3
model.add(Conv2D(128, 3, activation='relu', padding='same'))
model.add(Conv2D(128, 3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

# Fully connected layer
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

# Output layer
model.add(Dense(29, activation='softmax'))
model.summary()

# Early stopping, learning rate reduction and Best Model checkpoint
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=4, restore_best_weights=True, verbose=1)
reduce_learning_rate = ReduceLROnPlateau(monitor='val_accuracy', patience=2, factor=0.5, verbose=1)
checkpoint = ModelCheckpoint('models/asl/aslmodel_best.h5', monitor='val_loss', save_best_only=True, verbose=1)

# Model Compilation
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Training the Model
asl_class = model.fit(train_data, validation_data=val_data, epochs=30, callbacks=[early_stopping, reduce_learning_rate, checkpoint], verbose=1)

# Save the model
model.save('models/asl/aslmodel_final.h5')

#=======================================================================================================================

# Evaluation for train generator
loss, acc = model.evaluate(train_data, verbose=0)
print('The accuracy of the model for training data is:', acc * 100)
print('The Loss of the model for training data is:', loss)

# Evaluation for validation generator
loss, acc = model.evaluate(val_data, verbose=0)
print('The accuracy of the model for validation data is:', acc * 100)
print('The Loss of the model for validation data is:', loss)

# Evaluation on test set
loss, acc = model.evaluate(test_data, verbose=0)
print('The accuracy of the model for testing data is:', acc * 100)
print('The Loss of the model for testing data is:', loss)

# Prediction and Evaluation on the Test Set
result = model.predict(test_data, verbose=0)
y_pred = np.argmax(result, axis=1)
y_true = test_data.labels

# Classification Report
print(classification_report(y_true, y_pred, target_names=categories.values()))
 ```
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

import os
import numpy as np
import pandas as pd
import splitfolders
import tensorflow as tf

from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.models import Sequential
from keras.src.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
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

splitfolders.ratio('datasets/asl_main/asl_alphabet_train/asl_alphabet_train', output='workdir/', seed=1333, ratio=(0.8, 0.1, 0.1))

datagen = ImageDataGenerator(rescale=1.0 / 255)

train_path = 'workdir/train'
val_path = 'workdir/val'
test_path = 'workdir/test'

batch = 64
image_size = 200
img_channel = 3
n_classes = 36

train_data = datagen.flow_from_directory(directory=train_path, target_size=(image_size, image_size), batch_size=batch, class_mode='categorical')
val_data = datagen.flow_from_directory(directory=val_path, target_size=(image_size, image_size), batch_size=batch, class_mode='categorical')
test_data = datagen.flow_from_directory(directory=test_path, target_size=(image_size, image_size), batch_size=batch, class_mode='categorical', shuffle=False)


model = Sequential()
# Conv layer 1
model.add(Conv2D(32, 3, activation='relu', padding='same', input_shape=(image_size, image_size, img_channel)))
model.add(Conv2D(32, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(padding='same'))
model.add(Dropout(0.2))

# Conv layer 2
model.add(Conv2D(64, 3, activation='relu', padding='same'))
model.add(Conv2D(64, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(padding='same'))
model.add(Dropout(0.3))

# Conv layer 3
model.add(Conv2D(128, 3, activation='relu', padding='same'))
model.add(Conv2D(128, 3, activation='relu', padding='same'))
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

# Early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3, restore_best_weights=True, verbose=1)
reduce_learning_rate = ReduceLROnPlateau(monitor='val_accuracy', patience=2, factor=0.5, verbose=1)

# Model Compilation
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the Model
asl_class = model.fit(train_data, validation_data=val_data, epochs=30, callbacks=[early_stopping, reduce_learning_rate], verbose=1)

# Evaluation for train generator
loss, acc = model.evaluate(train_data, verbose=0)
print('The accuracy of the model for training data is:', acc * 100)
print('The Loss of the model for training data is:', loss)

# Evaluation for validation generator
loss, acc = model.evaluate(val_data, verbose=0)
print('The accuracy of the model for validation data is:', acc * 100)
print('The Loss of the model for validation data is:', loss)

# Prediction and Evaluation on the Test Set
result = model.predict(test_data, verbose=0)
y_pred = np.argmax(result, axis=1)
y_true = test_data.labels

# Evaluation on test set
loss, acc = model.evaluate(test_data, verbose=0)
print('The accuracy of the model for testing data is:', acc * 100)
print('The Loss of the model for testing data is:', loss)

# Classification Report
print(classification_report(y_true, y_pred, target_names=categories.values()))

# Save the model
model.save('models/asl/aslmodel.keras')

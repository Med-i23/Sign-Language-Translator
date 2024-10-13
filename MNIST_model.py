import pandas as pd
import numpy as np
import seaborn as sns
from keras.src.models import Sequential
from keras.src.layers import Dense, Conv2D, Flatten , Dropout , BatchNormalization
from keras.src.layers.pooling.max_pooling2d import MaxPooling2D
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.callbacks import ReduceLROnPlateau
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

train_df = pd.read_csv("datasets/MNIST/sign_mnist_train.csv")
test_df = pd.read_csv("datasets/MNIST/sign_mnist_test.csv")

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
        featurewise_center=False,  # set input mean to 0 over the datasets
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the datasets
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=30,  # randomly rotate images in the range
        zoom_range = 0.2, # Randomly zoom image
        width_shift_range=0.2,  # randomly shift images horizontally
        height_shift_range=0.2,  # randomly shift images vertically
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
model.save('models/MNIST/smnist.keras')


# ======================================================================================================================


# Evaluation
print("Accuracy of the model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")


# Classification report
y_pred = model.predict(x_test)
y_pred_classes = y_pred.argmax(axis=1)
y_true = y_test.argmax(axis=1)
print(classification_report(y_true, y_pred_classes))


# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g',
             xticklabels=label_binarizer.classes_,
             yticklabels=label_binarizer.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# Validation accuracy and loss
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Over Epochs')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Over Epochs')
plt.show()
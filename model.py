import pandas as pd

from keras.src.models import Sequential
from keras.src.layers import Dense, Conv2D, Flatten , Dropout , BatchNormalization
from keras.src.layers.pooling.max_pooling2d import MaxPooling2D
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import LabelBinarizer

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
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1, # randomly shift images vertically (fraction of total height)
        brightness_range=[0.8, 1.2],  # brightness adjusting
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)

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

# Flattening from 3D to 1D
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

# Train to model
history = model.fit(datagen.flow(x_train,y_train, batch_size = 128) ,epochs = 20 , validation_data = (x_test, y_test) , callbacks = [learning_rate_reduction])

# Save the trained model
model.save('models/smnist.keras')

print("Accuracy of the model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")


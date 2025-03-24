import os
import gdown
import zipfile
import numpy as np
import pandas as pd
import splitfolders
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


# ==========================Current model changes (to implent or already done)=====================================

# Debugging Order

# Check Data Leakage (set(train_files) & set(val_files))
# Fix Label Mapping (Duplicate "G" issue)
# Increase Data Augmentation (rotation_range=30)
# Adjust Learning Rate (1e-4 initially, then lower)
# Tune Dropout (0.6 or 0.7)
# Gradually Fine-Tune (freeze first 100 layers, then lower)
# Fix Class Weight Calculation
# Save Final Model Properly

#=Dataset-download=======================================================================================

file_id = "1M02Tyj-I3LLwPKse3QlUrjTHf7DHkjIf"
output_zip = "asl_dataset.zip"
output_dir = "datasets/asl_main/"

if not os.path.exists(output_dir):
    print("Dataset not found locally. Downloading from Google Drive...")

    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_zip, quiet=False)

    print("Extracting dataset...")
    with zipfile.ZipFile(output_zip, 'r') as zip_ref:
        zip_ref.extractall("datasets/")

    os.remove(output_zip)
    print("Complete.")
else:
    print("Dataset already exists. Skipping download.")

#=Training===============================================================================================
base_path = "../datasets/asl_main/asl_alphabet_train/asl_alphabet_train/"

# New folder for the model and all the data
base_model_dir = 'asl'
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
session_dir = os.path.join(base_model_dir, current_time)
os.makedirs(session_dir, exist_ok=True)

categories = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H",
    8: "I", 9: "J", 10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P",
    16: "Q", 17: "R", 18: "S", 19: "T", 20: "U", 21: "V", 22: "W", 23: "X",
    24: "Y", 25: "Z", 26: "del", 27: "nothing", 28: "space"
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
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='nearest'
)


train_path = '../workdir/train'
val_path = '../workdir/val'
test_path = '../workdir/test'

batch = 32
image_size = 200
img_channel = 3
n_classes = 29

train_data = datagen.flow_from_directory(directory=train_path, target_size=(image_size, image_size), batch_size=batch, class_mode='categorical')
val_data = datagen.flow_from_directory(directory=val_path, target_size=(image_size, image_size), batch_size=batch, class_mode='categorical')
test_data = datagen.flow_from_directory(directory=test_path, target_size=(image_size, image_size), batch_size=batch, class_mode='categorical', shuffle=False)

# ResNet model setup
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(image_size, image_size, img_channel))
base_model.trainable = True

for layer in base_model.layers[:-100]:
    layer.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.6),
    Dense(n_classes, activation='softmax')
])

# Model Compilation
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks
checkpoint_path = os.path.join(session_dir, 'resnet50_asl_best.h5')
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, restore_best_weights=True, verbose=1)
reduce_learning_rate = ReduceLROnPlateau(monitor='val_accuracy', patience=2, factor=0.5, verbose=1)
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)

# Class Weights
y_true = train_data.classes
class_weights = compute_class_weight('balanced', classes=np.unique(y_true), y=y_true)
class_weights = dict(enumerate(class_weights))

# Training
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,
    callbacks=[early_stopping, reduce_learning_rate, checkpoint],
    verbose=1,
    class_weight=class_weights
)

# Save the model
# final_model_path = os.path.join(session_dir, 'aslmodel_final.h5')
# model.save(session_dir, save_format='h5')

#=Evaluations=============================================================================================================

evaluation_dir = os.path.join(session_dir, 'evaluation')
os.makedirs(evaluation_dir, exist_ok=True)

train_loss, train_acc = model.evaluate(train_data, verbose=0)
val_loss, val_acc = model.evaluate(val_data, verbose=0)
test_loss, test_acc = model.evaluate(test_data, verbose=0)

# Evaluation results to a text file
evaluation_results_path = os.path.join(evaluation_dir, 'evaluation_results.txt')
with open(evaluation_results_path, 'w') as f:
    f.write(f"Training Data:\nAccuracy: {train_acc * 100:.2f}%\nLoss: {train_loss:.4f}\n\n")
    f.write(f"Validation Data:\nAccuracy: {val_acc * 100:.2f}%\nLoss: {val_loss:.4f}\n\n")
    f.write(f"Test Data:\nAccuracy: {test_acc * 100:.2f}%\nLoss: {test_loss:.4f}\n\n")

print(f"Evaluation results saved to: {evaluation_results_path}")

# Predict on the test set
result = model.predict(test_data, verbose=0)
y_pred = np.argmax(result, axis=1)
y_true = test_data.labels

# Classification report
classification_report_path = os.path.join(evaluation_dir, 'classification_report.txt')
classification_report_str = classification_report(y_true, y_pred, target_names=categories.values(), zero_division=0)
with open(classification_report_path, 'w') as f:
    f.write(classification_report_str)

print(f"Classification report saved to: {classification_report_path}")

# Confusion matrix plot
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories.values(), yticklabels=categories.values())
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')

confusion_matrix_path = os.path.join(evaluation_dir, 'confusion_matrix.png')
plt.savefig(confusion_matrix_path)
plt.close()

print(f"Confusion matrix saved to: {confusion_matrix_path}")

# Training history plots
# Accuracy plot
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
accuracy_plot_path = os.path.join(evaluation_dir, 'accuracy_plot.png')
plt.savefig(accuracy_plot_path)
plt.close()

# Loss plot
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
loss_plot_path = os.path.join(evaluation_dir, 'loss_plot.png')
plt.savefig(loss_plot_path)
plt.close()

print(f"Training history plots saved to: {evaluation_dir}")

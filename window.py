import os
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2 as cv
import mediapipe as mp
import numpy as np
import time
import pandas as pd
from keras.src.saving import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Sign Language Translator")
        self.geometry("1080x600")

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=0)
        self.grid_rowconfigure(3, weight=0)

        self.video_frame = tk.Frame(self, bg='lightgray')
        self.video_frame.grid(row=0, column=0, columnspan=2, sticky='nsew')

        self.video_label = tk.Label(self.video_frame, bg='lightgray')
        self.video_label.pack(fill=tk.BOTH, expand=True)

        self.button = tk.Button(self, text="Translate", command=self.on_button_click, font=("Arial", 14), bg='blue',
                                fg='white')
        self.button.grid(row=1, column=0, padx=10, pady=10, sticky='w')

        self.quit_button = tk.Button(self, text="Quit", command=self.quit_app, font=("Arial", 14), bg='red', fg='white')
        self.quit_button.grid(row=1, column=1, padx=10, pady=10, sticky='e')

        self.status_bar = tk.Label(self, text="Ready", bg='lightgray', anchor=tk.W)
        self.status_bar.grid(row=2, column=0, columnspan=2, sticky='we')

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )

        self.cap = cv.VideoCapture(0)
        self.video_width = 800
        self.video_height = 600

        self.model = load_model("models/smnist.keras")

        self.letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                           'T', 'U', 'V', 'W', 'X', 'Y']

        self.update_video()

    def on_button_click(self):
        success, frame = self.cap.read()
        if success:
            analysisframe = frame
            framergbanalysis = cv.cvtColor(analysisframe, cv.COLOR_BGR2RGB)
            resultanalysis = self.hands.process(framergbanalysis)
            hand_landmarksanalysis = resultanalysis.multi_hand_landmarks
            if hand_landmarksanalysis:
                h, w, _ = frame.shape
                for handLMsanalysis in hand_landmarksanalysis:
                    x_max, y_max = 0, 0
                    x_min, y_min = w, h
                    for lmanalysis in handLMsanalysis.landmark:
                        x, y = int(lmanalysis.x * w), int(lmanalysis.y * h)
                        x_min, x_max = min(x, x_min), max(x, x_max)
                        y_min, y_max = min(y, y_min), max(y, y_max)
                    y_min, y_max = y_min - 20, y_max + 20
                    x_min, x_max = x_min - 20, x_max + 20

                analysisframe = cv.cvtColor(analysisframe, cv.COLOR_BGR2GRAY)
                analysisframe = analysisframe[y_min:y_max, x_min:x_max]
                analysisframe = cv.resize(analysisframe, (28, 28))

                pixeldata = np.array(analysisframe).reshape(-1, 28, 28, 1) / 255.0
                prediction = self.model.predict(pixeldata)

                predarray = np.array(prediction[0])
                letter_prediction_dict = {self.letterpred[i]: predarray[i] for i in range(len(self.letterpred))}

                top_letter = max(letter_prediction_dict, key=letter_prediction_dict.get)
                top_confidence = letter_prediction_dict[top_letter]

                self.status_bar.config(text=f"Prediction: {top_letter}, Confidence: {100 * top_confidence:.2f}%")

                predarrayordered = sorted(predarray, reverse=True)
                high1, high2, high3 = predarrayordered[:3]
                for key, value in letter_prediction_dict.items():
                    if value == high1:
                        print(f"Predicted Character 1: {key}, Confidence: {100 * value:.2f}%")
                    elif value == high2:
                        print(f"Predicted Character 2: {key}, Confidence: {100 * value:.2f}%")
                    elif value == high3:
                        print(f"Predicted Character 3: {key}, Confidence: {100 * value:.2f}%")

    def update_video(self):
        success, frame = self.cap.read()
        if success:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frame = cv.resize(frame, (self.video_width, self.video_height))

            hands_detected = self.hands.process(frame)
            if hands_detected.multi_hand_landmarks:
                h, w, _ = frame.shape
                for hand_landmarks in hands_detected.multi_hand_landmarks:
                    x_max, y_max = 0, 0
                    x_min, y_min = w, h
                    for lm in hand_landmarks.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        x_min, x_max = min(x, x_min), max(x, x_max)
                        y_min, y_max = min(y, y_min), max(y, y_max)
                    y_min, y_max = y_min - 20, y_max + 20
                    x_min, x_max = x_min - 20, x_max + 20

                    cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

        self.after(10, self.update_video)

    def quit_app(self):
        self.cap.release()
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()

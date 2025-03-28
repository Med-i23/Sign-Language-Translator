import os
import tkinter as tk
import gdown
import cv2 as cv
import mediapipe as mp
import numpy as np
import random
import keras

from tkinter import ttk
from PIL import Image, ImageTk

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


#==Model-downloads======================================================================================================

file_ids = {
    "resnet50.h5": "1-dRJSRC401mvnm_TtCTBaTpq2ZZiCATd",
    "og_cnn.h5": "1IDKmRIDm9ocCMDNMoQDgDskkBoTleKMZ",
}

output_dir = "models/asl/"

os.makedirs(output_dir, exist_ok=True)

for filename, file_id in file_ids.items():
    output_path = os.path.join(output_dir, filename)

    if not os.path.exists(output_path):
        print(f"{filename} not found locally. Downloading from Google Drive...")

        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

    else:
        print(f"{filename} already exists. Skipping download.")

print("Downloads complete.")

#=App===================================================================================================================

model_path = "models/asl/resnet50.h5"

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Sign Language Translator")
        self.geometry("1080x700")

        # Notebook
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        # Tabs
        self.translator_frame = tk.Frame(self.notebook)
        self.learning_frame = tk.Frame(self.notebook)
        self.notebook.add(self.translator_frame, text="Translator")
        self.notebook.add(self.learning_frame, text="Learning")

        self.translator_tab()
        self.learning_tab()

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )

        self.cap = cv.VideoCapture(0)
        self.video_width = 800
        self.video_height = 600

        # Load whichever model you want
        self.model = keras.models.load_model(model_path)

        self.letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                           'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

        self.update_video()
        self.generate_random_letter()


    def translator_tab(self):
        self.translator_frame.grid_columnconfigure(0, weight=1)
        self.translator_frame.grid_columnconfigure(1, weight=1)
        self.translator_frame.grid_rowconfigure(0, weight=1)
        self.translator_frame.grid_rowconfigure(1, weight=0)
        self.translator_frame.grid_rowconfigure(2, weight=0)
        self.translator_frame.grid_rowconfigure(3, weight=0)

        # Video Frame
        self.video_frame = tk.Frame(self.translator_frame, bg='lightgray')
        self.video_frame.grid(row=0, column=0, columnspan=2, sticky='nsew')

        self.video_label = tk.Label(self.video_frame, bg='lightgray')
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Control Buttons
        self.translate_button = tk.Button(self.translator_frame, text="Translate", command=self.on_button_click,
                                          font=("Arial", 14), bg='blue', fg='white')
        self.translate_button.grid(row=1, column=0, padx=10, pady=10, sticky='w')

        # self.speak_button = tk.Button(self.translator_frame, text="Speak", command=self.speak,
        #                                   font=("Arial", 14), bg='blue', fg='white')
        # self.speak_button.grid(row=1, column=0, padx=10, pady=10, sticky='w')

        self.clear_button = tk.Button(self.translator_frame, text="Clear", command=self.clear_text,
                                      font=("Arial", 14), bg='orange', fg='white')
        self.clear_button.grid(row=1, column=0, padx=150, pady=10, sticky='w')

        self.quit_button = tk.Button(self.translator_frame, text="Quit", command=self.quit_app,
                                     font=("Arial", 14), bg='red', fg='white')
        self.quit_button.grid(row=1, column=1, padx=10, pady=10, sticky='e')

        self.status_bar = tk.Label(self.translator_frame, text="Ready", bg='lightgray', anchor=tk.W)
        self.status_bar.grid(row=2, column=0, columnspan=2, sticky='we')

        self.translated_text = tk.Text(self.translator_frame, height=2, font=("Arial", 18))
        self.translated_text.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky='we')

    def learning_tab(self):
        self.learning_frame.grid_columnconfigure(0, weight=1)
        self.learning_frame.grid_columnconfigure(1, weight=1)
        self.learning_frame.grid_rowconfigure(0, weight=1)
        self.learning_frame.grid_rowconfigure(1, weight=0)
        self.learning_frame.grid_rowconfigure(2, weight=0)

        self.learning_video_frame = tk.Frame(self.learning_frame, bg='lightgray')
        self.learning_video_frame.grid(row=0, column=0, columnspan=2, sticky='nsew')

        self.learning_video_label = tk.Label(self.learning_video_frame, bg='lightgray')
        self.learning_video_label.pack(fill=tk.BOTH, expand=True)

        self.letter_label = tk.Label(self.learning_frame, text="A", font=("Arial", 48), fg="black")
        self.letter_label.grid(row=1, column=0, padx=10, pady=10, columnspan=2)

        self.learning_translate_button = tk.Button(self.learning_frame, text="Translate", command=self.on_button_click,
                                                   font=("Arial", 14), bg='blue', fg='white')
        self.learning_translate_button.grid(row=2, column=0, padx=10, pady=10, sticky='w')

        self.quit_button_learning = tk.Button(self.learning_frame, text="Quit", command=self.quit_app,
                                              font=("Arial", 14), bg='red', fg='white')
        self.quit_button_learning.grid(row=2, column=1, padx=10, pady=10, sticky='e')


    # TODO
    def speak(self):
        pass

    def generate_random_letter(self):
        self.current_letter = random.choice(self.letterpred)
        self.letter_label.config(text=self.current_letter)


    def flash_letter(self, color):
        self.letter_label.config(fg=color, font=("Arial", 48, "bold"))
        self.after(500, self.reset_letter)

    def reset_letter(self):
        self.letter_label.config(fg="black", font=("Arial", 48))


    def on_button_click(self):
        success, frame = self.cap.read()
        if success:
            analysis_frame = frame
            frame_rgb_analysis = cv.cvtColor(analysis_frame, cv.COLOR_BGR2RGB)
            result_analysis = self.hands.process(frame_rgb_analysis)
            hand_landmarks_analysis = result_analysis.multi_hand_landmarks

            if hand_landmarks_analysis:
                h, w, _ = frame.shape
                for handLMs_analysis in hand_landmarks_analysis:
                    x_max, y_max = 0, 0
                    x_min, y_min = w, h
                    for lmanalysis in handLMs_analysis.landmark:
                        x, y = int(lmanalysis.x * w), int(lmanalysis.y * h)
                        x_min, x_max = min(x, x_min), max(x, x_max)
                        y_min, y_max = min(y, y_min), max(y, y_max)

                    padding = 50
                    y_min, y_max = max(0, y_min - padding), min(h, y_max + padding)
                    x_min, x_max = max(0, x_min - padding), min(w, x_max + padding)

                analysis_frame = frame_rgb_analysis[y_min:y_max, x_min:x_max]
                analysis_frame = cv.resize(analysis_frame, (200, 200))

               # cv.imshow("Cropped Hand Region", cv.cvtColor(analysis_frame, cv.COLOR_RGB2BGR))

                pixel_data = np.array(analysis_frame).reshape(-1, 200, 200, 3) / 255.0

                try:
                    prediction = self.model.predict(pixel_data)
                except Exception as e:
                    self.status_bar.config(text=f"Error in prediction: {str(e)}")
                    return

                pred_array = np.array(prediction[0])
                letter_prediction_dict = {self.letterpred[i]: pred_array[i] for i in range(len(self.letterpred))}

                top_letter = max(letter_prediction_dict, key=letter_prediction_dict.get)
                top_confidence = letter_prediction_dict[top_letter]

                self.status_bar.config(text=f"Prediction: {top_letter}, Confidence: {100 * top_confidence:.2f}%")

                self.translated_text.insert(tk.END, top_letter)

                predarrayordered = sorted(pred_array, reverse=True)
                high1, high2, high3 = predarrayordered[:3]
                for key, value in letter_prediction_dict.items():
                    if value == high1:
                        print(f"Predicted Character 1: {key}, Confidence: {100 * value:.2f}%")
                    elif value == high2:
                        print(f"Predicted Character 2: {key}, Confidence: {100 * value:.2f}%")
                    elif value == high3:
                        print(f"Predicted Character 3: {key}, Confidence: {100 * value:.2f}%")

                if top_letter == self.current_letter:
                    self.flash_letter('green')
                    self.after(1000, self.generate_random_letter)
                else:
                    self.flash_letter('red')
            else:
                self.translated_text.insert(tk.END, " ")
                self.status_bar.config(text="SPACE")

    def clear_text(self):
        self.translated_text.delete(1.0, tk.END)

    def update_video(self):
        success, frame = self.cap.read()
        if success:
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frame_resized = cv.resize(frame_rgb, (self.video_width, self.video_height))

            hands_detected = self.hands.process(frame_resized)
            if hands_detected.multi_hand_landmarks:
                h, w, _ = frame_resized.shape
                for hand_landmarks in hands_detected.multi_hand_landmarks:
                    x_max, y_max = 0, 0
                    x_min, y_min = w, h
                    for lm in hand_landmarks.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        x_min, x_max = min(x, x_min), max(x, x_max)
                        y_min, y_max = min(y, y_min), max(y, y_max)
                    y_min, y_max = y_min - 20, y_max + 20
                    x_min, x_max = x_min - 20, x_max + 20

                    cv.rectangle(frame_resized, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            img = Image.fromarray(frame_resized)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

            img_learning = Image.fromarray(frame_resized)
            imgtk_learning = ImageTk.PhotoImage(image=img_learning)
            self.learning_video_label.imgtk = imgtk_learning
            self.learning_video_label.config(image=imgtk_learning)

        self.after(10, self.update_video)

    def clear_text(self):
        self.translated_text.delete(1.0, tk.END)

    def quit_app(self):
        self.cap.release()
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()

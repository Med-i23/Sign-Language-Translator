import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2 as cv
import mediapipe as mp

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Sign Language Translator")
        self.geometry("1080x600")

        # Configure grid columns and rows
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=0)
        self.grid_rowconfigure(3, weight=0)

        # Create a frame for the video feed
        self.video_frame = tk.Frame(self, bg='lightgray')
        self.video_frame.grid(row=0, column=0, columnspan=2, sticky='nsew')

        # Label for displaying the video feed
        self.video_label = tk.Label(self.video_frame, bg='lightgray')
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Action button
        self.button = tk.Button(self, text="Translate", command=self.on_button_click, font=("Arial", 14), bg='blue', fg='white')
        self.button.grid(row=1, column=0, padx=10, pady=10, sticky='w')

        # Quit button
        self.quit_button = tk.Button(self, text="Quit", command=self.quit_app, font=("Arial", 14), bg='red', fg='white')
        self.quit_button.grid(row=1, column=1, padx=10, pady=10, sticky='e')

        # Status bar
        self.status_bar = tk.Label(self, text="Ready", bg='lightgray', anchor=tk.W)
        self.status_bar.grid(row=2, column=0, columnspan=2, sticky='we')

        # Setup Mediapipe Hands
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        # Start the video capture
        self.cap = cv.VideoCapture(0)

        # Start the video loop
        self.update_video()

    def on_button_click(self):
        messagebox.showinfo("Info", "Translate button clicked!")
        self.status_bar.config(text="Translating...")

    def quit_app(self):
        self.cap.release()  # Release the camera when quitting
        self.destroy()

    def update_video(self):
        success, frame = self.cap.read()

        if success:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            hands_detected = self.hands.process(frame)

            if hands_detected.multi_hand_landmarks:
                for hand_landmarks in hands_detected.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        self.mp_styles.get_default_hand_landmarks_style(),
                        self.mp_styles.get_default_hand_connections_style()
                    )

            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

        self.after(10, self.update_video)

if __name__ == "__main__":
    app = App()
    app.mainloop()

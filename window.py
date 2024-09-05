import tkinter as tk
from tkinter import messagebox


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("My Python App")
        self.geometry("900x600")

        self.label = tk.Label(self, text="Sign Language Translator", font=("Arial", 18))
        self.label.pack(pady=20)

        self.button = tk.Button(self, text="Action", command=self.on_button_click)
        self.button.pack(pady=10)

        self.quit_button = tk.Button(self, text="Quit", command=self.quit_app)
        self.quit_button.pack(pady=10)

    def on_button_click(self):
        messagebox.showinfo("Info", "You clicked the button!")

    def quit_app(self):
        self.quit()


if __name__ == "__main__":
    app = App()
    app.mainloop()

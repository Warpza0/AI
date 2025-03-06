from tkinter import Tk, Label, Button, StringVar, Frame
import torch
from train import train_model  # Assuming train.py contains a function to train the model

class MNISTApp:
    def __init__(self, master):
        self.master = master
        master.title("MNIST Digit Classifier")

        self.frame = Frame(master)
        self.frame.pack(padx=10, pady=10)

        self.label = Label(self.frame, text="MNIST Digit Classifier", font=("Helvetica", 16))
        self.label.pack()

        self.status = StringVar()
        self.status_label = Label(self.frame, textvariable=self.status, font=("Helvetica", 12))
        self.status_label.pack(pady=(10, 0))

        self.train_button = Button(self.frame, text="Train Model", command=self.train_model)
        self.train_button.pack(pady=(10, 0))

        self.quit_button = Button(self.frame, text="Quit", command=master.quit)
        self.quit_button.pack(pady=(10, 0))

    def train_model(self):
        self.status.set("Training in progress...")
        self.master.update()  # Update the GUI to show the status

        # Call the training function from train.py
        train_model()

        self.status.set("Training complete!")

if __name__ == "__main__":
    root = Tk()
    app = MNISTApp(root)
    root.mainloop()
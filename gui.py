from tkinter import Tk, Label, Button, StringVar, Frame, filedialog
import torch
from train import train_model, load_image, load_text_file  # Assuming train.py contains these functions

torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MNISTApp:
    def __init__(self, master):
        self.master = master
        master.title("MNIST Digit Classifier")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.frame = Frame(master)
        self.frame.pack(padx=10, pady=10)

        self.label = Label(self.frame, text="MNIST Digit Classifier", font=("Helvetica", 16))
        self.label.pack()

        self.status = StringVar()
        self.status_label = Label(self.frame, textvariable=self.status, font=("Helvetica", 12))
        self.status_label.pack(pady=(10, 0))

        self.train_button = Button(self.frame, text="Train Model", command=self.train_model)
        self.train_button.pack(pady=(10, 0))

        self.load_image_button = Button(self.frame, text="Load Image", command=self.load_image, state='disabled')
        self.load_image_button.pack(pady=(10, 0))

        self.load_text_button = Button(self.frame, text="Load Text File", command=self.load_text_file)
        self.load_text_button.pack(pady=(10, 0))

        self.quit_button = Button(self.frame, text="Quit", command=master.quit)
        self.quit_button.pack(pady=(10, 0))

        self.model = None

    def train_model(self):
        self.status.set("Training in progress...")
        self.master.update()  # Update the GUI to show the status

        # Call the training function from train.py
        self.model = train_model(self.device)

        self.status.set("Training complete!")
        self.load_image_button.config(state='normal')  # Enable the load image button after training

    def load_image(self):
        if self.model is None:
            self.status.set("Please train the model first.")
            return

        file_path = filedialog.askopenfilename()
        if not file_path:
            return

        self.status.set("Loading image...")
        self.master.update()

        image_tensor = load_image(file_path)
        image_tensor = image_tensor.to(self.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(image_tensor)
            predicted_class = output.argmax(dim=1).item()

        self.status.set(f"Predicted class: {predicted_class}")

    def load_text_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if not file_path:
            return

        self.status.set("Loading text file...")
        self.master.update()

        text_content = load_text_file(file_path)
        self.status.set(f"Text file content: {text_content[:100]}...")  # Display first 100 characters

if __name__ == "__main__":
    root = Tk()
    app = MNISTApp(root)
    root.mainloop()

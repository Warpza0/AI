from tkinter import Tk, Label, Button, StringVar, Frame, Entry, Text, Scrollbar, filedialog, END
import torch
from train import train_model, load_image, load_text_file  # Assuming train.py contains these functions

class MNISTApp:
    def __init__(self, master):
        self.master = master
        master.title("Chatbot")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.frame = Frame(master)
        self.frame.pack(padx=10, pady=10)

        self.label = Label(self.frame, text="placeholder", font=("Malgun Gothic Semilight", 16))
        self.label.pack()

        self.label = Label(self.frame, text= "In all cases if you can't resolve the issue contact your IT buddy first, then Marianna, then bluesys", font=("Malgun Gothic Semilight", 16))
        self.label.pack()

        self.status = StringVar()
        self.status_label = Label(self.frame, textvariable=self.status, font=("Malgun Gothic Semilight", 12))
        self.status_label.pack(pady=(10, 0))

        self.chat_label = Label(self.frame, text="Ask placeholder a question:", font=("Malgun Gothic Semilight", 12))
        self.chat_label.pack(pady=(10, 0))

        self.chat_entry = Entry(self.frame, width=50)
        self.chat_entry.pack(pady=(5, 0))

        self.chat_button = Button(self.frame, text="Ask", font=("Malgun Gothic Semilight", 12), command=self.ask_question)
        self.chat_button.pack(pady=(5, 0))

        self.chat_response = Text(self.frame, height=10, width=50, state='disabled', wrap='word')
        self.chat_response.pack(pady=(10, 0))

        self.quit_button = Button(self.frame, text="Quit", font=("Malgun Gothic Semilight", 12), command=master.quit)
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

    def ask_question(self):
        question = self.chat_entry.get()
        if not question.strip():
            return

        # Basic rule-based AI logic for a support bot
        if "vpn" in question.lower():
            response = "follow the process of restarting your laptop, if password has been recently changed contact bluesys." 
        elif "laptop locked" in question.lower():
            response = "If your laptop account is locked, please contact Bluesys for help."
        elif "camera not working" in question.lower():
            response = "check your teams camera settings then restart teams, if the issue persists restart your laptop."
        elif "splash top" in question.lower():
            response = "Have your IT buddy force close the task in task manager."
        elif "printer" in question.lower():
            response = "search for new devices on windows, add new printer."
        elif "emails slow" "emails crashing" in question.lower():
            response = "restart outlook first, if that doesn't work restart your laptop."
        else:
            response = "I'm sorry, I don't understand your question. Please try asking something else."

        # Display the response in the chat response box
        self.chat_response.config(state='normal')
        self.chat_response.insert(END, f"User: {question}\n")
        self.chat_response.insert(END, f"placeholder: {response}\n\n")
        self.chat_response.config(state='disabled')
        self.chat_entry.delete(0, END)

if __name__ == "__main__":
    root = Tk()
    app = MNISTApp(root)
    root.mainloop()

from tkinter import Tk, Label, Button, StringVar, Frame, Entry, Text, filedialog, END
import torch
from train import train_model, load_image, load_text_file  # Assuming train.py contains these functions

class MNISTApp:
    def __init__(self, master):
        # Initialize the main window
        self.master = master
        master.title("Technical Assistant Chatbot")

        # Set the device (CPU or GPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create the main frame
        self.frame = Frame(master)
        self.frame.pack(padx=10, pady=10)

        # Title label
        self.label = Label(self.frame, text="Technical Assistant Chatbot", font=("Malgun Gothic Semilight", 16))
        self.label.pack()

        # Instruction label
        self.label = Label(
            self.frame,
            text="If you can't resolve the issue, contact your IT buddy first, then Marianna, then Bluesys.",
            font=("Malgun Gothic Semilight", 12),
            wraplength=400
        )
        self.label.pack(pady=(10, 0))

        # Status label for dynamic updates
        self.status = StringVar()
        self.status_label = Label(self.frame, textvariable=self.status, font=("Malgun Gothic Semilight", 12))
        self.status_label.pack(pady=(10, 0))

        # Chat input label
        self.chat_label = Label(self.frame, text="Ask the chatbot a question:", font=("Malgun Gothic Semilight", 12))
        self.chat_label.pack(pady=(10, 0))

        # Chat input field
        self.chat_entry = Entry(self.frame, width=50)
        self.chat_entry.pack(pady=(5, 0))

        # Submit button for asking questions
        self.chat_button = Button(self.frame, text="Ask", font=("Malgun Gothic Semilight", 12), command=self.ask_question)
        self.chat_button.pack(pady=(5, 0))

        # Chat response area
        self.chat_response = Text(self.frame, height=10, width=50, state='disabled', wrap='word')
        self.chat_response.pack(pady=(10, 0))

        # Quit button to close the application
        self.quit_button = Button(self.frame, text="Finish", font=("Malgun Gothic Semilight", 12), command=master.quit)
        self.quit_button.pack(pady=(10, 0))

        # Placeholder for the trained model
        self.model = None

    def ask_question(self):
        """Handles user input and provides a response."""
        question = self.chat_entry.get().strip()  # Get and clean user input
        if not question:
            return  # Do nothing if the input is empty

        # Get the chatbot's response
        response = self.get_response(question)

        # Display the conversation in the chat response area
        self.chat_response.config(state='normal')  # Enable the text area for editing
        self.chat_response.insert(END, f"User: {question}\n")
        self.chat_response.insert(END, f"Chatbot: {response}\n\n")
        self.chat_response.config(state='disabled')  # Disable the text area to prevent user edits
        self.chat_entry.delete(0, END)  # Clear the input field

    def get_response(self, question):
        """Generates a response based on the user's question."""
        # Rule-based logic for predefined responses
        question_lower = question.lower()
        if "vpn" in question_lower:
            return "Follow the process of restarting your laptop. If your password has been recently changed, contact Bluesys."
        elif "laptop locked" in question_lower:
            return "If your laptop account is locked, please contact Bluesys for help."
        elif "camera not working" in question_lower:
            return "Check your Teams camera settings, then restart Teams. If the issue persists, restart your laptop."
        elif "splash top" in question_lower:
            return "Have your IT buddy force close the task in Task Manager."
        elif "printer" in question_lower:
            return "Search for new devices on Windows and add a new printer."
        elif "emails slow" in question_lower or "slow emails" in question_lower:
            return "Restart Outlook first. If that doesn't work, restart your laptop."
        elif "io" in question_lower:
            return "Contact the Ops team. They will raise it with IO."
        elif "hi" in question_lower or "hello" in question_lower:
            return "Hello! Welcome to your technical assistant. How can I help you?"
        else:
            return "I'm sorry, I don't understand your question. Please try asking something else."

if __name__ == "__main__":
    # Create the main application window
    root = Tk()
    app = MNISTApp(root)
    root.mainloop()

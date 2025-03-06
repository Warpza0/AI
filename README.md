# AI Project

This project is what i'm hoping to create a rival of chatgpt with, currently at a base stage with more work coming over the future.

## Project Structure

```
AI
|
├── main.py        # Entry point for the application
├── gui.py         # Implementation of the graphical user interface
├── train.py       # Training logic for the neural network
|── utils.py       # Utility functions for data preprocessing and evaluation
├── requirements.txt    # List of dependencies
└── README.md           # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd AI
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the application, execute the following command:
```
python AI/main.py
```

This will launch the GUI, where you can start training the neural network on the MNIST dataset. The interface will display the training loss over epochs and provide options to visualize the results.

## Functionality

- **Training the Model**: The application allows users to initiate the training process, during which the model's performance is displayed in real-time.
- **Visualizing Loss**: Users can see a plot of the training loss over epochs, helping them understand how well the model is learning.
- **Checking Accuracy**: After training, the application provides an option to check the accuracy of the model on the training dataset.

## Dependencies

The project requires the following libraries:
- PyTorch
- Tkinter or PyQt (for GUI)
- Matplotlib (for plotting)
- torchvision (for dataset handling)

Ensure that you have the appropriate versions of these libraries installed as specified in `requirements.txt`.

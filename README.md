# MNIST GUI Project

This project implements a graphical user interface (GUI) for training a neural network on the MNIST dataset using PyTorch. The application allows users to visualize the training process and check the accuracy of the model on the training data.

## Project Structure

```
mnist-gui-project
├── src
│   ├── main.py        # Entry point for the application
│   ├── gui.py         # Implementation of the graphical user interface
│   ├── train.py       # Training logic for the neural network
│   └── utils.py       # Utility functions for data preprocessing and evaluation
├── requirements.txt    # List of dependencies
└── README.md           # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd mnist-gui-project
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the application, execute the following command:
```
python src/main.py
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
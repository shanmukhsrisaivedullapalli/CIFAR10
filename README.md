# CIFAR-10 Image Classification

This project demonstrates the process of building, training, and evaluating both an Artificial Neural Network (ANN) and a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset.

## Project Overview

The CIFAR-10 dataset is a popular dataset for machine learning and computer vision tasks. It consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 test images.

This project includes:

- Data preprocessing and normalization
- Visualization of sample images
- Building and training an ANN
- Building and training a CNN
- Comparing the performance of the ANN and CNN models
- Saving the models for future use
- Predicting the class of a new image using the trained models

## Getting Started

### Prerequisites

To run this project, you'll need:

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- PIL (Python Imaging Library)

You can install the necessary libraries using pip:

```bash
pip install tensorflow numpy matplotlib pillow
```

### Running the Code

1. **Clone the repository:**

   ```bash
   git clone https://github.com/shanmukhsrisaivedullapalli/CIFAR10.git
   cd CIFAR10
   ```

2. **Run the script:**

   Open the cifar10_classification.ipynb file in Jupyter Notebook or Jupyter Lab to execute the code cells step-by-step.

3. **Evaluate the models:**

   After training, the script will display the training accuracy and loss for both the ANN and CNN models. It will also evaluate the CNN model on the test dataset.

4. **Make predictions:**

   The script includes a function to preprocess and predict the class of a new image using the trained CNN model.

### Example Output

Here are some example results:

- **ANN Model:** Achieves around 69.52% accuracy on the testing data.
- **CNN Model:** Achieves around 97.53% accuracy on the training data and 69.52% on the testing data.

## File Structure

- `cifar10_classification.py`: Main script containing the entire workflow.
- `ann_model.pkl`: Pickle file containing the trained ANN model.
- `cnn_model.pkl`: Pickle file containing the trained CNN model.
- `README.md`: Project overview and instructions.

## Future Work

- Experiment with different architectures and hyperparameters to improve accuracy.
- Implement data augmentation techniques to further boost performance.
- Deploy the model as a web application using Flask or Django.

# Handwritten Digit Recognition using CNN
# Project Overview

This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify handwritten digits from the MNIST dataset.
The model is trained to recognize digits from 0 to 9 with high accuracy.

# Dataset

The project uses the built-in MNIST dataset provided by Keras.

Dataset contains:

60,000 training images
10,000 testing images
Grayscale images of size 28 × 28 pixels

# Technologies Used
Python
TensorFlow
Keras
NumPy
Matplotlib
Seaborn
Scikit-learn
CNN Architecture

The CNN model contains:

Convolution Layer (32 filters)
Max Pooling Layer
Convolution Layer (64 filters)
Max Pooling Layer
Flatten Layer
Dense Hidden Layer
Output Layer with Softmax Activation

# Data Preprocessing

The images are normalized by dividing pixel values by 255.

X_train = X_train / 255
X_test = X_test / 255

The data is reshaped for CNN input:

X_train_reshaped = X_train.reshape(-1, 28, 28, 1)
X_test_reshaped = X_test.reshape(-1, 28, 28, 1)

# Model Compilation
cnn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Training

The model is trained using:

model_fit = cnn_model.fit(
    X_train_reshaped,
    y_train,
    epochs=5,
    validation_split=0.2
)

# Model Performance
Training Accuracy
Approximately 95%
Validation Accuracy
Approximately 96%
Test Accuracy
96.36%

# Prediction

The trained model predicts handwritten digits using:

y_pred = cnn_model.predict(X_test_reshaped)

The predicted class is obtained using:

predicted = np.argmax(y_pred[0])
Visualizations

The project includes:

Sample digit visualization
Accuracy graph
Loss graph
Confusion matrix heatmap

# Confusion Matrix

A confusion matrix is generated to evaluate prediction performance for each digit class.

cm = confusion_matrix(y_test, y_pred_labels)

# Heatmap visualization:

sns.heatmap(cm, annot=True, fmt='d')
Conclusion

This project demonstrates how CNNs can effectively classify handwritten digits with high accuracy.
The model successfully learns image features such as edges and patterns using convolution and pooling operations.

The project is a beginner-friendly implementation of Deep Learning and Computer Vision concepts using TensorFlow and Keras.

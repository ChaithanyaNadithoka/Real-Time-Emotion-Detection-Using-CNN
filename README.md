# Real-Time Emotion Detection using CNN

A deep learning project using a Convolutional Neural Network (CNN) to detect facial emotions in real-time from a webcam feed. The model is trained on the **FER2013** dataset and uses OpenCV for real-time face detection.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## 📖 Overview

This project implements a real-time emotion detection system capable of recognizing **seven emotions**:
- Angry 😡  
- Disgusted 🤢  
- Fearful 😨  
- Happy 😃  
- Neutral 😐  
- Sad 😢  
- Surprised 😮  

Using a webcam, it captures live video, detects faces using a **Haar Cascade Classifier**, and predicts emotions using a CNN model trained on the FER2013 dataset. The system overlays the detected emotion on the video in real-time.

---

## 🚀 Features

- **Real-Time Emotion Detection** using OpenCV and a pre-trained CNN.
- **Accurate Classification** into seven emotion categories.
- **Model Visualization** with accuracy and loss plots during training.
- **Efficient Training** using TensorFlow and Keras.
- **User-Friendly Modes**:
  - `train`: Train the model on the FER2013 dataset.
  - `display`: Perform real-time emotion detection.

---

## 🗂 Dataset

- **FER2013**:  
  - A publicly available dataset for facial expression recognition.
  - Consists of **35,887 grayscale images** sized **48x48 pixels**.
  - Divided into **7 emotion classes**.
- Download it from [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).

---

## 🧠 Model Architecture

The CNN model consists of the following layers:
- **Convolutional Layers:** Extract features using 3x3 kernels.
- **ReLU Activation:** Adds non-linearity to the model.
- **MaxPooling Layers:** Reduces dimensions and prevents overfitting.
- **Dropout Layers:** Randomly drops neurons to enhance generalization.
- **Flatten Layer:** Converts feature maps into a vector.
- **Fully Connected Layers:** Perform classification using the **softmax** function.

**Loss Function:** Categorical Cross-Entropy  
**Optimizer:** Adam with a learning rate of **0.0001**  

---

## 📊 Results

- After training, the model achieves competitive accuracy on the validation set.
- The generated `plot.png` will visualize the accuracy and loss during training.

---

## 🤝 Contributing

Contributions are welcome!

- Fork the repository
- Create a new branch (`feature/my-feature`)
- Commit your changes
- Push to your branch
- Open a Pull Request

Please ensure your changes follow the code style guidelines.

---

## 🛡 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for more details.

---

## 💡 Acknowledgements

- [Kaggle](https://www.kaggle.com/) for providing the FER2013 dataset.
- [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) for the deep learning framework.
- [OpenCV](https://opencv.org/) for real-time face detection.


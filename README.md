# Stress detection using multimodal physiological Data
A machine learning and deep learning–based system for detecting human stress levels using multiple physiological signals such as heart rate, skin conductance, temperature, and respiration.
## Project Overview
Stress significantly affects both physical and mental health. Traditional stress detection methods rely on self-reporting, which is subjective and often unreliable. This project proposes an automated stress detection system using multimodal physiological data combined with advanced machine learning and deep learning models.
By analyzing body signals in real time, the system aims to classify stress levels accurately and support applications in healthcare, workplace wellness, and human–computer interaction.

## Objectives

Collect and process multimodal physiological signals
Extract meaningful features from biosignals
Build machine learning and deep learning models for stress classification
Improve accuracy using multimodal data fusion
Evaluate system performance using standard metrics

## Physiological Signals Used
| Signal                       | Description                    | Stress Indicator                   |
| ---------------------------- | ------------------------------ | ---------------------------------- |
| ECG / Heart Rate (HR)        | Measures heart activity        | Stress increases heart rate        |
| GSR (Galvanic Skin Response) | Skin conductivity due to sweat | Higher during stress               |
| Skin Temperature (ST)        | Peripheral body temperature    | Often drops under stress           |
| Respiration Rate (RR)        | Breathing pattern              | Faster and irregular during stress |

# system architecture 
## Data Acquisition
Sensors collect physiological signals from subjects.
## Preprocessing
Noise filtering
Signal normalization
Segmentation into time windows
## Feature Extraction
Time-domain features (mean, variance, peak rate)
Frequency-domain features (power spectral density)
Statistical features

## Multimodal Data Fusion
Features from different signals are combined into a single feature vector.
## Model Training
####  Models used:

- Machine Learning: SVM, Random Forest, KNN

- Deep Learning: CNN, LSTM

## Stress Classification
####  Output classes:

- Relaxed

- Mild Stress

- High Stress

## Machine Learning and Deep Learning Models
### Traditional Machine Learning Models

- Support Vector Machine (SVM)

- Random Forest Classifier

- K-Nearest Neighbors (KNN)

### Deep Learning Models

- Convolutional Neural Network (CNN) for spatial feature extraction

- Long Short-Term Memory (LSTM) networks for temporal pattern learning in physiological signals

## Dataset

The dataset contains synchronized physiological signals recorded under different stress conditions.

#### Data Includes:

- Heart Rate

- GSR

- Skin Temperature

- Respiration Rate

- Stress labels (Relaxed / Stressed)

The dataset may be obtained from public sources (such as WESAD) or collected using wearable sensors.

## Technologies Used

- Programming Language: Python

- Libraries:

- NumPy

- Pandas

- Scikit-learn

- TensorFlow / Keras / PyTorch

- Matplotlib / Seaborn

## Evaluation Metrics

The system performance is evaluated using:

- Accuracy

- Precision

- Recall

- F1-Score

- Confusion Matrix

##  to Run the Project
###  Clone the repository
### Navigate into the project folder
cd stress-detection-multimodal
### Install dependencies
pip install -r requirements.txt
### Run training script
python train_model.py
### Run testing
python test_model.py

## Applications

- Healthcare stress monitoring

- Workplace mental wellness systems

- Adaptive gaming based on emotional state

- Driver stress monitoring

- Mental health research

  ## Future Improvements

- Real-time stress detection using wearable devices

- Mobile application integration

- Personalized stress models

- Adding more biosignals (EEG, EMG)

- Edge AI deployment for on-device processing

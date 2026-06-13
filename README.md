# Advanced Cyber Attack Detection System Using Federated Learning

## Overview

The Advanced Cyber Attack Detection System is a machine learning-based Intrusion Detection System (IDS) developed using Federated Learning and a Hybrid CNN-BiLSTM deep learning model. The project is designed to detect malicious network traffic and cyber attacks while preserving data privacy across multiple clients.

Instead of sharing raw data with a central server, each client trains the model locally and only model parameters are shared, making the system more secure and privacy-preserving.

---

## Features

* Federated Learning Architecture
* Hybrid CNN + BiLSTM Deep Learning Model
* Cyber Attack Detection
* Interactive Streamlit Dashboard
* Dataset Visualization
* Class Distribution Analysis
* Missing Value Analysis
* Confusion Matrix Visualization
* Classification Report
* Training Accuracy and Loss Graphs
* Real-time Performance Metrics

---

## Technologies Used

* Python
* Streamlit
* TensorFlow / Keras
* Scikit-Learn
* Pandas
* NumPy
* Matplotlib
* Seaborn

---

## Dataset

This project uses the CICIDS2017 dataset.

Datasets used:

* Friday-WorkingHours-Afternoon-DDos
* Friday-WorkingHours-Afternoon-PortScan
* Wednesday-workingHours

The network traffic is classified into:

* BENIGN Traffic
* ATTACK Traffic

---

## Model Architecture

The system uses a Hybrid CNN-BiLSTM architecture:

1. Conv1D Layer
2. Batch Normalization
3. Max Pooling Layer
4. Bidirectional LSTM Layer
5. Dropout Layer
6. Dense Layer
7. Output Layer (Sigmoid)

---

## Federated Learning Workflow

1. Load client datasets.
2. Preprocess data locally.
3. Train local models on each client.
4. Aggregate model updates using Federated Averaging.
5. Update the global model.
6. Evaluate global model performance.
7. Display results through Streamlit Dashboard.

---

## Performance

Achieved Results:

* Accuracy: 99%+
* Precision: High Attack Detection Performance
* Recall: Effective Threat Identification
* F1-Score: Balanced Detection Capability

Note: Results may vary depending on dataset selection and training configuration.

---

## Dashboard Modules

### Dataset Statistics

Displays the number of records available for each client.

### Dataset Distribution

Visualizes dataset sizes across multiple clients.

### Class Distribution

Shows the distribution of BENIGN and ATTACK classes.

### Missing Value Analysis

Identifies missing values within datasets.

### Federated Training

Performs federated model training and displays training progress.

### Performance Evaluation

Displays:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report

---

## Future Enhancements

* Real-time Network Traffic Monitoring
* Multi-Class Attack Classification
* Explainable AI (XAI) Integration
* Deployment on Cloud Infrastructure
* Edge Device Integration
* Secure Federated Aggregation

---

## Conclusion

This project demonstrates how Federated Learning can be combined with Deep Learning techniques to build a privacy-preserving and highly accurate Cyber Attack Detection System. The solution helps organizations detect network intrusions while keeping sensitive data decentralized and secure.

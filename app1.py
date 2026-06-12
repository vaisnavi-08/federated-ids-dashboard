# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 12:00:46 2026

@author: VAISNAVI
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    BatchNormalization,
    Dropout,
    Bidirectional,
    LSTM,
    Dense,
    MaxPooling1D
)
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)
# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Advanced Cyber Attack Detection System",
    layout="wide"
)

st.title("🛡️ Federated Learning Intrusion Detection System")
st.markdown("---")

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Upload Client Datasets")

client1_file = st.sidebar.file_uploader(
    "Upload Client 1 CSV",
    type=["csv"],
    key="client1"
)

client2_file = st.sidebar.file_uploader(
    "Upload Client 2 CSV",
    type=["csv"],
    key="client2"
)

client3_file = st.sidebar.file_uploader(
    "Upload Client 3 CSV",
    type=["csv"],
    key="client3"
)

# =====================================================
# STORAGE
# =====================================================

missing_percentages = {}

# =====================================================
# PREPROCESS FUNCTION
# =====================================================

def preprocess_dataset(file, client_name):

    df = pd.read_csv(file)

    # Sample for RAM optimization
    df = df.sample(
        n=min(50000, len(df)),
        random_state=42
    )

    # Detect label column
    label_col = [c for c in df.columns if "Label" in c][0]

    df.rename(
        columns={label_col: "Label"},
        inplace=True
    )
    df = df.sample(
    n=min(50000, len(df)),
    random_state=42
    )
    # Convert labels
    df["Label"] = df["Label"].apply(
        lambda x: 0 if str(x).strip() == "BENIGN" else 1
    )

    # Replace Inf
    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    # Missing %
    missing_percent = (
        df.isnull().sum() / len(df)
    ) * 100

    missing_percentages[client_name] = missing_percent

    # Fill Missing Values
    df = df.fillna(
        df.median(numeric_only=True)
    )

    # Features and Labels
    X = df.drop("Label", axis=1)
    y = df["Label"]

    scaler = StandardScaler()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = X_train.reshape(
    X_train.shape[0],
    2,
    39
    )

    X_test = X_test.reshape(
    X_test.shape[0],
    2,
    39
    )

    return (
    X_train,
    X_test,
    y_train,
    y_test,
    df
    )
# =====================================================
# PROCESS DATASETS
# =====================================================

def hybrid_cnn_bilstm(input_shape):

    model = Sequential([

        Conv1D(
            128,
            kernel_size=3,
            activation='relu',
            padding='same',
            input_shape=input_shape
        ),

        BatchNormalization(),

        MaxPooling1D(
            pool_size=2,
            padding='same'
        ),

        Bidirectional(
            LSTM(
                64,
                return_sequences=False
            )
        ),

        Dropout(0.3),

        Dense(
            64,
            activation='relu'
        ),

        Dropout(0.3),

        Dense(
            1,
            activation='sigmoid'
        )

    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

if (
    client1_file is not None and
    client2_file is not None and
    client3_file is not None
):

    with st.spinner("Processing datasets..."):

        X1_train, X1_test, y1_train, y1_test, df1 = preprocess_dataset(
            client1_file,
            "Client1"
        )

        X2_train, X2_test, y2_train, y2_test, df2 = preprocess_dataset(
            client2_file,
            "Client2"
        )

        X3_train, X3_test, y3_train, y3_test, df3 = preprocess_dataset(
            client3_file,
            "Client3"
        )

    st.success("Datasets Loaded Successfully")

    # =====================================================
    # DATASET STATS
    # =====================================================

    st.header("📊 Dataset Statistics")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Client 1 Records",
        len(df1)
    )

    c2.metric(
        "Client 2 Records",
        len(df2)
    )

    c3.metric(
        "Client 3 Records",
        len(df3)
    )

    # =====================================================
    # DATASET DISTRIBUTION
    # =====================================================

    st.subheader("Client Dataset Distribution")

    fig, ax = plt.subplots(figsize=(8,4))

    sizes = [
        len(df1),
        len(df2),
        len(df3)
    ]

    ax.bar(
        ["Client1", "Client2", "Client3"],
        sizes
    )

    ax.set_ylabel("Records")
    ax.set_title("Client Dataset Sizes")

    st.pyplot(fig)

    # =====================================================
    # CLASS DISTRIBUTION
    # =====================================================

    st.subheader("Class Distribution")

    col1, col2, col3 = st.columns(3)

    datasets = [
        (df1, "Client1"),
        (df2, "Client2"),
        (df3, "Client3")
    ]

    for col, (df_temp, title) in zip(
        [col1, col2, col3],
        datasets
    ):

        fig, ax = plt.subplots(figsize=(4,3))

        label_counts = df_temp["Label"].map({
            0: "BENIGN",
            1: "ATTACK"
        }).value_counts()

        label_counts.plot(
            kind="bar",
            ax=ax
        )

        ax.set_title(title)
        ax.set_ylabel("Count")

        col.pyplot(fig)

    # =====================================================
    # MISSING VALUE ANALYSIS
    # =====================================================

    st.subheader("Missing Value Analysis")

    selected_client = st.selectbox(
        "Choose Client",
        ["Client1", "Client2", "Client3"],
        key="missing_client"
    )

    miss = missing_percentages[selected_client]

    miss = miss[miss > 0]

    if len(miss) > 0:

        fig, ax = plt.subplots(figsize=(10,5))

        miss.sort_values(
            ascending=False
        ).plot(
            kind="bar",
            ax=ax
        )

        ax.set_ylabel("Missing Percentage")

        st.pyplot(fig)

    else:

        st.success("No Missing Values Found")

    # =====================================================
    # DATA PREVIEW
    # =====================================================

    st.subheader("Dataset Preview")

    preview_client = st.selectbox(
        "Choose Dataset",
        ["Client1", "Client2", "Client3"],
        key="preview_client"
    )

    if preview_client == "Client1":
        st.dataframe(df1.head())

    elif preview_client == "Client2":
        st.dataframe(df2.head())

    else:
        st.dataframe(df3.head())
#dunction_weigted_average

def weighted_aggregate_weights(
    client_weights,
    client_sizes
):

    total_samples = np.sum(client_sizes)

    weighted_avg = []

    for weights_tuple in zip(*client_weights):

        layer_avg = np.sum(
            [
                client_sizes[i] *
                np.array(weights_tuple[i])
                for i in range(
                    len(client_weights)
                )
            ],
            axis=0
        ) / total_samples

        weighted_avg.append(
            layer_avg
        )

    return weighted_avg        
        
# ==========================================
# FEDERATED TRAINING
# ==========================================

st.markdown("---")
st.header("🚀 Federated Training")

start_training = st.button(
    "Start Federated Training"
)

if start_training:

    st.success("Training Started...")

    clients = {
        "Client1": (X1_train, y1_train),
        "Client2": (X2_train, y2_train),
        "Client3": (X3_train, y3_train)
    }

    X_test_global = np.concatenate(
        [X1_test, X2_test, X3_test],
        axis=0
    )

    y_test_global = pd.concat(
        [y1_test, y2_test, y3_test]
    )
    X_train_global = np.concatenate(
    [X1_train, X2_train, X3_train],
    axis=0
)

    y_train_global = pd.concat(
        [y1_train, y2_train, y3_train]
)
    input_shape = (
        X1_train.shape[1],
        X1_train.shape[2]
    )

    global_model = hybrid_cnn_bilstm(
        input_shape
    )

    progress_bar = st.progress(0)

    status = st.empty()

    acc_history = []
    loss_history = []

    rounds = 15

    for r in range(rounds):

        status.write(
            f"Training Round {r+1}/{rounds}"
        )
        history = global_model.fit(
            X_train_global,
            y_train_global,
            validation_data=(
                  X_test_global,
                  y_test_global
            ),
            epochs=1,
            batch_size=256,
            verbose=0
           )
        

        train_acc = history.history["accuracy"][0]
        train_loss = history.history["loss"][0]

        acc_history.append(train_acc)
        loss_history.append(train_loss)

        progress_bar.progress(
            int((r + 1) * 100 / rounds)
        )

    loss, acc = global_model.evaluate(
        X_test_global,
        y_test_global,
        verbose=0
    )
    

    # Save trained model
    global_model.save("federated_ids_model.h5")
    
    with open("federated_ids_model.h5", "rb") as file:
               st.download_button(
                     "📥 Download Trained Model",
                     file,

                     file_name="federated_ids_model.h5"
     )
    
    

    st.success(
    f"Training Complete! Accuracy = {acc:.4f}"
      )

    st.success("Model saved successfully as federated_ids_model.h5")
    
    #predictions
    
    y_pred_prob = global_model.predict(
    X_test_global,
    verbose=0
    )

    y_pred = (
        y_pred_prob > 0.5
        ).astype(int).flatten()
    
    
    #metrics dashboad
    
    precision = precision_score(
    y_test_global,
    y_pred
    )

    recall = recall_score(
       y_test_global,
       y_pred
     )

    f1 = f1_score(
       y_test_global,
       y_pred
     )

    m1, m2, m3 = st.columns(3)

    m1.metric(
         "Precision",
         f"{precision:.4f}"
     )

    m2.metric(
         "Recall",
         f"{recall:.4f}"
     )

    m3.metric(
        "F1 Score",
        f"{f1:.4f}"
     )
    
    
    #confusion matrix
    
    st.subheader(
    "📊 Confusion Matrix"
    )
    
    y_test_global = np.array(
          y_test_global
          ).flatten()

    cm = confusion_matrix(
       y_test_global,
       y_pred
     )

    fig, ax = plt.subplots(figsize=(6,4))

    
    sns.heatmap(
              cm,
              annot=True,
              fmt="d",
              cmap="Blues",
              ax=ax
            )

    ax.set_xlabel(
         "Predicted"
    )

    ax.set_ylabel(
         "Actual"
    )

    st.pyplot(fig)
    
    #classification report
    
    st.subheader("📄 Classification Report")

    report = classification_report(
         y_test_global,
         y_pred,
         output_dict=True
     )

    st.dataframe(
          pd.DataFrame(report).transpose()
     )
    
    
    
    
    
    # ==========================
    # Metrics
    # ==========================

    c1, c2 = st.columns(2)

    c1.metric(
        "Final Accuracy",
        f"{acc:.4f}"
     )

    c2.metric(
        "Final Loss",
        f"{loss:.4f}"
     )

    # ==========================
    # Accuracy Graph
    # ==========================

    st.subheader("📈 Training Accuracy")
    fig, ax = plt.subplots(figsize=(8,4))

    ax.plot(
    range(1, len(acc_history) + 1),
    acc_history,
    marker="o"
    )

    ax.grid(True)

    ax.set_xlabel("Federated Round")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Federated Round")

    st.pyplot(fig)

    # ==========================
    # Loss Graph
    # ==========================

    st.subheader(
          "📉 Training Loss"
    )

    fig, ax = plt.subplots()

    ax.plot(
    range(1, len(loss_history) + 1),
    loss_history,
    marker="o"
    )

    ax.set_xlabel("Round")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs Round")

    st.pyplot(fig)


else:

      st.info("⬅ Upload all 3 CSV files from the sidebar to continue.")
    
    

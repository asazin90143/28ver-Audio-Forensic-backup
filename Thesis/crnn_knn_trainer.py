import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, f1_score
import json

# Set random seed for reproducibility in results
np.random.seed(42)
tf.random.set_seed(42)

def generate_synthetic_data(num_samples=1000, num_classes=5, time_steps=128, features=64):
    """Generates synthetic audio features (e.g., Log-Mel Spectrograms) and labels."""
    X = np.random.randn(num_samples, time_steps, features, 1)
    
    # Inject class-specific patterns to make it learnable
    y = np.random.randint(0, num_classes, num_samples)
    for i in range(num_samples):
        c = y[i]
        # Add a sine wave pattern specific to the class
        freq = (c + 1) * 5
        t = np.linspace(0, 2 * np.pi, time_steps)
        pattern = np.sin(freq * t)[:, np.newaxis]
        X[i, :, :, 0] += pattern * 0.5

    return X, y

def build_crnn_model(input_shape, num_classes):
    """Builds the CRNN feature extractor."""
    model = models.Sequential()
    
    # CNN Blocks
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Reshape for RNN
    # Current shape: (batch, time_steps/8, features/8, 128)
    model.add(layers.Reshape((-1, 128)))
    
    # RNN Block (LSTM)
    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=False)))
    model.add(layers.Dropout(0.3))
    
    # Embedding Layer (Output used for KNN)
    model.add(layers.Dense(64, activation='relu', name='embedding'))
    
    # Classification head (used just for training the representations)
    model.add(layers.Dense(num_classes, activation='softmax', name='classification'))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    print("--- Starting CRNN-KNN Training Pipeline ---")
    classes = ["Gunshot", "Siren", "Glass Breaking", "Scream", "Human Voice"]
    num_classes = len(classes)
    
    # 1. Generate Data
    print("Generating synthetic dataset...")
    X_train, y_train = generate_synthetic_data(num_samples=800, num_classes=num_classes)
    X_val, y_val = generate_synthetic_data(num_samples=200, num_classes=num_classes)
    
    # 2. Build and Train CRNN
    print("Building and training CRNN model...")
    model = build_crnn_model(input_shape=(128, 64, 1), num_classes=num_classes)
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=40,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Extract Feature Extractor
    feature_extractor = models.Model(inputs=model.inputs, outputs=model.get_layer('embedding').output)

    
    # 3. Train KNN on Embeddings
    print("Extracting embeddings for KNN...")
    train_embeddings = feature_extractor.predict(X_train)
    val_embeddings = feature_extractor.predict(X_val)
    
    print("Training KNN classifier...")
    knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    knn.fit(train_embeddings, y_train)
    
    # 4. Evaluate KNN
    y_pred = knn.predict(val_embeddings)
    report = classification_report(y_val, y_pred, target_names=classes, output_dict=True)
    
    print("\nClassification Report (KNN):")
    for cls in classes:
        print(f"{cls}: Precision={report[cls]['precision']:.2f}, Recall={report[cls]['recall']:.2f}, F1={report[cls]['f1-score']:.2f}")
    
    # 5. Generate Outputs
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Save Loss Graph
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('CRNN Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_path = os.path.join(output_dir, 'training_loss_curve.png')
    plt.savefig(loss_path)
    print(f"Saved loss curve to {loss_path}")
    
    # Save Metrics JSON
    metrics = {cls: {"Precision": round(report[cls]['precision']*100), 
                     "Recall": round(report[cls]['recall']*100), 
                     "F1-Score": round(report[cls]['f1-score']*100)} 
               for cls in classes}
    
    metrics['Average'] = {
        "Precision": round(report['macro avg']['precision']*100),
        "Recall": round(report['macro avg']['recall']*100),
        "F1-Score": round(report['macro avg']['f1-score']*100)
    }
    
    metrics_path = os.path.join(output_dir, 'event_detection_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {metrics_path}")
    print("CRNN-KNN pipeline completed successfully.")

if __name__ == "__main__":
    main()

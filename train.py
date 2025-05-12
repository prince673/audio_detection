import argparse
import os
import numpy as np
from tensorflow.keras.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    ReduceLROnPlateau
)
from data_loader import WaveFakeLoader
from features import AudioFeatureExtractor
from model import DeepfakeDetector
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to WaveFake dataset directory")
    parser.add_argument("--model_type", type=str, default="cnn",
                       choices=["cnn", "cnn_lstm", "mlp"],
                       help="Model architecture to use")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training")
    args = parser.parse_args()
    
    # Load and preprocess data
    print("Loading dataset...")
    loader = WaveFakeLoader(data_path=args.data_path)
    X_train, X_test, y_train, y_test = loader.load_dataset()
    
    # Feature extraction
    feature_extractor = AudioFeatureExtractor()
    
    if args.model_type == "cnn":
        print("Extracting mel spectrograms for CNN...")
        X_train = np.array([feature_extractor.extract_mel_spectrogram(x) for x in X_train])
        X_test = np.array([feature_extractor.extract_mel_spectrogram(x) for x in X_test])
        X_train = X_train[..., np.newaxis]  # Add channel dimension
        X_test = X_test[..., np.newaxis]
        model = DeepfakeDetector.create_cnn_model(X_train[0].shape)
        
    elif args.model_type == "cnn_lstm":
        print("Preparing CNN-LSTM input...")
        # Extract spectrograms for each 1-second segment
        segment_length = 16000  # 1 second at 16kHz
        segments_per_sample = 4  # For 4-second samples
        
        def create_segments(audio):
            spectrograms = []
            for i in range(segments_per_sample):
                segment = audio[i*segment_length:(i+1)*segment_length]
                if len(segment) < segment_length:
                    segment = np.pad(segment, (0, segment_length - len(segment)))
                spec = feature_extractor.extract_mel_spectrogram(segment)
                spectrograms.append(spec)
            return np.array(spectrograms)
        
        X_train = np.array([create_segments(x) for x in X_train])
        X_test = np.array([create_segments(x) for x in X_test])
        X_train = X_train[..., np.newaxis]
        X_test = X_test[..., np.newaxis]
        model = DeepfakeDetector.create_cnn_lstm_model(X_train[0].shape)
    
    else:  # MLP
        print("Extracting features for MLP...")
        X_train = np.array([feature_extractor.extract_features(x) for x in X_train])
        X_test = np.array([feature_extractor.extract_features(x) for x in X_test])
        model = DeepfakeDetector.create_mlp_model(X_train.shape[1])
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            "models/best_model.h5",
            save_best_only=True,
            monitor="val_accuracy",
            mode="max"
        ),
        EarlyStopping(
            patience=10,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ]
    
    # Train the model
    print(f"Training {args.model_type} model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    
    # Save the final model
    model.save("models/final_model.h5")
    print("Model saved to models/final_model.h5")

if __name__ == "__main__":
    main()
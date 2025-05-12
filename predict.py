import argparse
import numpy as np
import librosa
from features import AudioFeatureExtractor
from tensorflow.keras.models import load_model

class DeepfakeAudioDetector:
    def __init__(self, model_path, model_type="cnn", sample_rate=16000):
        self.model = load_model(model_path)
        self.feature_extractor = AudioFeatureExtractor(sample_rate=sample_rate)
        self.model_type = model_type
        self.sample_rate = sample_rate
        
    def predict(self, audio_path, duration=4):
        """Predict whether audio is real or fake"""
        try:
            # Load audio file
            audio, _ = librosa.load(audio_path, sr=self.sample_rate, duration=duration)
            
            # Pad if shorter than duration
            if len(audio) < duration * self.sample_rate:
                audio = np.pad(audio, (0, duration * self.sample_rate - len(audio)))
            
            # Extract features based on model type
            if self.model_type == "cnn":
                features = self.feature_extractor.extract_mel_spectrogram(audio)
                features = features[..., np.newaxis]
                features = np.expand_dims(features, axis=0)
            elif self.model_type == "cnn_lstm":
                segment_length = self.sample_rate  # 1 second segments
                segments = []
                for i in range(duration):  # For each second
                    segment = audio[i*segment_length:(i+1)*segment_length]
                    if len(segment) < segment_length:
                        segment = np.pad(segment, (0, segment_length - len(segment)))
                    spec = self.feature_extractor.extract_mel_spectrogram(segment)
                    segments.append(spec)
                features = np.array(segments)[np.newaxis, ..., np.newaxis]
            else:  # MLP
                features = self.feature_extractor.extract_features(audio)
                features = np.expand_dims(features, axis=0)
            
            # Make prediction
            prediction = self.model.predict(features)
            class_idx = np.argmax(prediction)
            confidence = prediction[0][class_idx]
            
            return {
                "prediction": "fake" if class_idx == 1 else "real",
                "confidence": float(confidence),
                "class_probabilities": {
                    "real": float(prediction[0][0]),
                    "fake": float(prediction[0][1])
                }
            }
        except Exception as e:
            return {"error": str(e)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, required=True,
                       help="Path to audio file for prediction")
    parser.add_argument("--model", type=str, default="models/final_model.h5",
                       help="Path to trained model")
    parser.add_argument("--model_type", type=str, default="cnn",
                       choices=["cnn", "cnn_lstm", "mlp"],
                       help="Type of model being used")
    args = parser.parse_args()
    
    detector = DeepfakeAudioDetector(args.model, args.model_type)
    result = detector.predict(args.audio)
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print("\nDeepfake Audio Detection Result:")
        print(f"Prediction: {result['prediction'].upper()}")
        print(f"Confidence: {result['confidence']*100:.2f}%")
        print(f"Real Probability: {result['class_probabilities']['real']*100:.2f}%")
        print(f"Fake Probability: {result['class_probabilities']['fake']*100:.2f}%")
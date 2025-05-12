import numpy as np
import librosa
from python_speech_features import mfcc, logfbank

class AudioFeatureExtractor:
    def __init__(self, sample_rate=16000, n_mfcc=40, n_fft=2048, hop_length=512):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def extract_features(self, audio):
        """Extract comprehensive audio features"""
        features = {}
        
        # Time-domain features
        features["zcr"] = librosa.feature.zero_crossing_rate(audio)
        features["rms"] = librosa.feature.rms(y=audio)
        
        # Frequency-domain features
        stft = np.abs(librosa.stft(audio))
        features["chroma"] = librosa.feature.chroma_stft(S=stft, sr=self.sample_rate)
        features["mel"] = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate)
        features["contrast"] = librosa.feature.spectral_contrast(S=stft, sr=self.sample_rate)
        features["tonnetz"] = librosa.feature.tonnetz(y=audio, sr=self.sample_rate)
        
        # MFCCs
        features["mfcc"] = librosa.feature.mfcc(
            y=audio, 
            sr=self.sample_rate, 
            n_mfcc=self.n_mfcc
        )
        
        # Combine all features
        combined = []
        for feature_name, feature_value in features.items():
            if len(feature_value.shape) > 1:  # For 2D features
                combined.extend(np.mean(feature_value, axis=1))
                combined.extend(np.std(feature_value, axis=1))
            else:  # For 1D features
                combined.append(np.mean(feature_value))
                combined.append(np.std(feature_value))
        
        return np.array(combined)
    
    def extract_mel_spectrogram(self, audio):
        """Extract mel spectrogram for CNN input"""
        S = librosa.feature.melspectrogram(
            y=audio, 
            sr=self.sample_rate, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length
        )
        S_dB = librosa.power_to_db(S, ref=np.max)
        return S_dB
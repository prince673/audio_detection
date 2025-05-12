import os
import numpy as np
import librosa
from tqdm import tqdm
from pydub import AudioSegment
import random

class WaveFakeLoader:
    def __init__(self, data_path, sample_rate=16000, duration=4, test_size=0.2):
        """
        Initialize WaveFake data loader
        
        Args:
            data_path: Path to WaveFake dataset
            sample_rate: Target sample rate
            duration: Duration of audio segments in seconds
            test_size: Fraction of data for testing
        """
        self.data_path = data_path
        self.sample_rate = sample_rate
        self.duration = duration
        self.test_size = test_size
        self.classes = ["real", "fake"]
        
    def load_dataset(self):
        """Load and preprocess the dataset"""
        features = []
        labels = []
        
        # Real audio (LJ Speech)
        real_path = os.path.join(self.data_path, "LJSpeech-1.1/wavs")
        print("Loading real audio samples...")
        real_features = self._load_audio_files(real_path, label=0)
        features.extend(real_features)
        labels.extend([0] * len(real_features))
        
        # Fake audio (WaveFake generated)
        fake_folders = ["melgan", "hifigan", "waveglow"]  # WaveFake subfolders
        for folder in fake_folders:
            fake_path = os.path.join(self.data_path, folder)
            if os.path.exists(fake_path):
                print(f"Loading fake audio samples from {folder}...")
                fake_features = self._load_audio_files(fake_path, label=1)
                features.extend(fake_features)
                labels.extend([1] * len(fake_features))
        
        # Convert to numpy arrays
        features = np.array(features)
        labels = np.array(labels)
        
        # Split into train/test
        return self._train_test_split(features, labels)
    
    def _load_audio_files(self, path, label):
        """Load audio files from directory and split into fixed-length segments"""
        features = []
        files = [f for f in os.listdir(path) if f.endswith(".wav")]
        
        # Take a subset if too many files
        if len(files) > 5000:  # Limit to 5000 files per class
            files = random.sample(files, 5000)
            
        for file in tqdm(files):
            try:
                audio, _ = librosa.load(os.path.join(path, file), 
                                     sr=self.sample_rate, 
                                     duration=self.duration)
                # Pad if shorter than duration
                if len(audio) < self.duration * self.sample_rate:
                    audio = np.pad(audio, (0, self.duration * self.sample_rate - len(audio)))
                features.append(audio)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        return features
    
    def _train_test_split(self, features, labels):
        """Split data into train and test sets"""
        from sklearn.model_selection import train_test_split
        return train_test_split(
            features, labels, 
            test_size=self.test_size, 
            random_state=42, 
            stratify=labels
        )
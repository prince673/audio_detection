from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, 
    Dense, Dropout, BatchNormalization,
    LSTM, TimeDistributed, Reshape
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

class DeepfakeDetector:
    @staticmethod
    def create_cnn_model(input_shape, num_classes=2):
        """CNN model for spectrogram analysis"""
        model = Sequential([
            # First Conv Block
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape,
                  kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Second Conv Block
            Conv2D(64, (3, 3), activation='relu', 
                  kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Third Conv Block
            Conv2D(128, (3, 3), activation='relu', 
                  kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Flatten and Dense Layers
            Flatten(),
            Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    @staticmethod
    def create_cnn_lstm_model(input_shape, num_classes=2):
        """Hybrid CNN-LSTM model for temporal analysis"""
        model = Sequential([
            # TimeDistributed CNN for frame analysis
            TimeDistributed(Conv2D(32, (3, 3), activation='relu'), 
                          input_shape=input_shape),
            TimeDistributed(BatchNormalization()),
            TimeDistributed(MaxPooling2D((2, 2))),
            TimeDistributed(Dropout(0.25)),
            
            TimeDistributed(Conv2D(64, (3, 3), activation='relu')),
            TimeDistributed(BatchNormalization()),
            TimeDistributed(MaxPooling2D((2, 2))),
            TimeDistributed(Dropout(0.25)),
            
            TimeDistributed(Flatten()),
            
            # LSTM for temporal dependencies
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.3),
            
            # Output layer
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
import json
import os
import time

import tensorflow as tf
from keras.src.utils.image_dataset_utils import load_image
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.keras.callbacks import EarlyStopping

from .constants import SEED, MODEL_PATH, MODEL_METRICS_PATH
from .data import Data
from .predict_item import PredictItem


class Model:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.last_updated = time.time()
        self.threshold = 0.3
        self.f1 = 0
        self.precision = 0
        self.recall = 0
        # Check if MODEL_PATH exists
        if not os.path.exists(MODEL_PATH):
            self.model = None
            self.trained = False
        else:
            print("Pre-trained model found. Loading model...")
            self.model = tf.keras.models.load_model(MODEL_PATH)
            self.trained = True
            if os.path.exists(MODEL_METRICS_PATH):
                metrics = json.load(open(MODEL_METRICS_PATH))
                self.f1 = metrics['f1']
                self.precision = metrics['precision']
                self.recall = metrics['recall']
                self.last_updated = metrics['last_updated']
            print("Model loaded.")
        self.data = None
        tf.random.set_seed(SEED)
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_validate = None
        self.y_validate = None
        self.load_data()
        
    def load_data(self):
        self.data = Data(self.dataset_path)
        self.X, self.y = self.data.separate_data(self.data.preprocessed_df)
        self.X_train, self.y_train, self.X_test, self.y_test, self.X_validate, self.y_validate = self.data.split_data(self.X, self.y)
        self.X_train, self.y_train = self.data.oversample_data(self.X_train, self.y_train)
    
    def train(self):
        self.load_data()
        
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Input((self.X_train.shape[1],)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ])

        early_stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=0.001,
            patience=10,
            restore_best_weights=True
        )

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(self.X_train, self.y_train, validation_data=(self.X_validate, self.y_validate),
                       callbacks=[early_stopping],
                       batch_size=8, epochs=60, verbose=2)
        
        self.model.save(MODEL_PATH)

        y_pred = self.model.predict(self.X_test)
        y_pred = (y_pred > 0.3).astype(int)
        self.f1 = f1_score(self.y_test, y_pred)
        self.precision = precision_score(self.y_test, y_pred)
        self.recall = recall_score(self.y_test, y_pred)
        
        # Saving metrics
        with open(MODEL_METRICS_PATH, 'w') as f:
            json.dump({'f1': self.f1, 'precision': self.precision, 'recall': self.recall, 'last_updated': self.last_updated}, f)
            f.close()

        self.last_updated = time.time()
        self.trained = True

        return self.f1, self.precision, self.recall

    def predict(self, predict_item: PredictItem):
        df = self.data.preprocess_predict_input(predict_item)
        return self.model.predict(df).item() > self.threshold

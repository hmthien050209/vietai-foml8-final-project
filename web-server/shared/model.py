import time

import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.keras.callbacks import EarlyStopping

from .constants import SEED
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
        self.model = None
        self.trained = False
        self.data = None
        tf.random.set_seed(SEED)

    def train(self):
        self.data = Data(self.dataset_path)
        X, y = self.data.separate_data(self.data.preprocessed_df)
        X_train, y_train, X_test, y_test, X_validate, y_validate = self.data.split_data(X, y)
        X_train, y_train = self.data.oversample_data(X_train, y_train)

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Input((X_train.shape[1],)),
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
        self.model.fit(X_train, y_train, validation_data=(X_validate, y_validate),
                       callbacks=[early_stopping],
                       batch_size=8, epochs=60, verbose=2)

        y_pred = self.model.predict(X_test)
        y_pred = (y_pred > 0.3).astype(int)
        self.f1 = f1_score(y_test, y_pred)
        self.precision = precision_score(y_test, y_pred)
        self.recall = recall_score(y_test, y_pred)

        self.last_updated = time.time()
        self.trained = True

        return self.f1, self.precision, self.recall

    def predict(self, predict_item: PredictItem):
        df = self.data.preprocess_predict_input(predict_item)
        return self.model.predict(df).item() > self.threshold

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
class RealtimePredictor:
    def __init__(self, model_path, sequence_length=10):
        self.sequence_length = sequence_length # sliding window size in dataset loader
        self.buffer = []  # cumulate recent datas
        self.model = tf.keras.models.load_model(model_path)
        self.scaler =  StandardScaler()
        print("model loaded")

    def update(self, headpose, gaze, blink):
        """
        get new data -> add to sequence buffer -> if buffer is full, predict(predict_if_ready)
        """
        self.buffer.append([headpose, gaze, blink])
        if len(self.buffer) > self.sequence_length:
            self.buffer.pop(0)

    def predict_if_ready(self):
        """
        if buffer has enough number of datas, predict the next class
        """
        if len(self.buffer) < self.sequence_length:
            return None, None

        input_seq = np.array(self.buffer).reshape(1, self.sequence_length, 3)
        # Scaling
        input_seq = self.scaler.fit_transform(input_seq.reshape(-1, input_seq.shape[-1])).reshape(input_seq.shape)
        probs = self.model.predict(input_seq, verbose=0)[0]
        pred_class = int(np.argmax(probs))

        return pred_class

    def reset(self):
        """
        reset the sequence buffer
        """
        self.buffer = []

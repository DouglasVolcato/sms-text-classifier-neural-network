from tensorflow.keras.preprocessing.sequence import pad_sequences
from Utils.message_classification import MessageClassification
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import tensorflow as tf


class TextClassifierModel:
    def predict(self, text: str) -> str:
        model = tf.keras.models.load_model(
            "cache/models/text_classifier_model.keras")

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts([text])

        sequence = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=100,
                               padding='post', truncating='post')

        prediction = model.predict(padded)

        if prediction > 0.5:
            return "ham"
        else:
            return "spam"

    def train(self) -> None:
        tokenizer, max_length, train_padded, train_dataset, test_padded, test_dataset = self.__getDataset()
        model = self.__getModel(tokenizer, max_length)

        model.fit(train_padded, train_dataset["v1"], epochs=10, validation_data=(
            test_padded, test_dataset["v1"]))

        loss, accuracy = model.evaluate(test_padded, test_dataset["v1"])
        print(f"Test accuracy: {accuracy:.4f}")

        model.save("cache/models/text_classifier_model.keras")

    def __getDataset(self) -> pd.DataFrame:
        dataset = pd.read_csv("cache/data/spam.csv", encoding="latin-1")
        dataset = dataset[['v1', 'v2']]
        dataset["v1"] = dataset["v1"].apply(MessageClassification.encode)

        train_dataset = dataset.sample(frac=0.8, random_state=0)
        test_dataset = dataset.drop(train_dataset.index)

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(train_dataset["v2"])

        max_length = 100
        train_sequences = tokenizer.texts_to_sequences(train_dataset["v2"])
        train_padded = pad_sequences(
            train_sequences, maxlen=max_length, padding='post', truncating='post')

        test_sequences = tokenizer.texts_to_sequences(test_dataset["v2"])
        test_padded = pad_sequences(
            test_sequences, maxlen=max_length, padding='post', truncating='post')

        return tokenizer, max_length, train_padded, train_dataset, test_padded, test_dataset

    def __getModel(self, tokenizer: Tokenizer, max_length: int) -> tf.keras.Sequential:
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=len(
                tokenizer.word_index) + 1, output_dim=64, input_length=max_length),  # Embedding layer
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])

        model.compile(optimizer="adam",
                      loss="binary_crossentropy", metrics=["accuracy"])

        return model

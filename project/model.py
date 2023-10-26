# TODO: modify this custom model to your liking. Check out this tutorial for more on this class: https://outerbounds.com/docs/nlp-tutorial-L2/
# TODO: train the model on traindf.
# TODO: score the model on valdf with _the same_ 2D metric space you used in previous cell.
# TODO: test your model works by importing the model module in notebook cells, and trying to fit traindf and score predictions on the valdf data!

from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers, regularizers
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OrdinalEncoder

from utils.metrics import calc_scores


class NbowModel:
    def __init__(self, vocab_sz_rev: int, vocab_sz_title: Optional[int] = 50):
        self.vocab_sz_rev = vocab_sz_rev
        self.vocab_sz_title = vocab_sz_title

        # Instantiate the CountVectorizer
        self.cv_reviews = CountVectorizer(
            min_df=0.005,
            max_df=0.75,
            stop_words="english",
            strip_accents="ascii",
            max_features=self.vocab_sz_rev,
        )
        self.cv_title = CountVectorizer(
            min_df=0.005,
            max_df=0.75,
            stop_words="english",
            strip_accents="ascii",
            max_features=self.vocab_sz_title,
        )

        self.enc_dep1 = CountVectorizer(
            min_df=0.005,
            max_df=0.75,
            stop_words="english",
            strip_accents="ascii",
            max_features=10,
        )
        self.enc_dep2 = CountVectorizer(
            min_df=0.005,
            max_df=0.75,
            stop_words="english",
            strip_accents="ascii",
            max_features=10,
        )
        self.enc_dep3 = CountVectorizer(
            min_df=0.005,
            max_df=0.75,
            stop_words="english",
            strip_accents="ascii",
            max_features=10,
        )

        #               age + reviews           + title             + departments
        self.input_size = 1 + self.vocab_sz_rev + self.vocab_sz_title + 18

        self.to_transform = [(self.cv_title, "title"),
                            (self.cv_reviews, "review"),
                            (self.enc_dep1, "division_name"),
                            (self.enc_dep2, "department_name"),
                            (self.enc_dep3, "class_name"),]

        # Define the keras model
        inputs = tf.keras.Input(shape=(self.input_size,), name="input")
        x = layers.Dropout(0.10)(inputs)
        x = layers.Dense(
            15,
            activation="relu",
            kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        )(x)
        x = layers.Dense(
            10,
            activation="relu",
            kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        )(x)
        predictions = layers.Dense(
            1,
            activation="sigmoid",
        )(x)
        self.model = tf.keras.Model(inputs, predictions)
        opt = optimizers.Adam(learning_rate=0.002)
        self.model.compile(
            loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"]
        )

    def transform(self, X):
        res = [trans.transform(X[col]) for trans, col in self.to_transform] + [X["age"].to_numpy()]
        res = [r.toarray() if len(np.shape(r)) == 2 else np.expand_dims(r, 1) for r in res]
        res = np.hstack(res)
        return res
    
    def fit_transform(self, X):
        [trans.fit(X[col]) for trans, col in self.to_transform] + [X["age"].to_numpy()]

    def fit(self, X, y):
        print(X.shape)
        self.fit_transform(X)
        res = self.transform(X)
        self.model.fit(x=res, y=y, batch_size=32, epochs=10, validation_split=0.2)

    def predict(self, X):
        print(X.shape)
        res = self.transform(X)
        return self.model.predict(res)

    def eval_acc(self, X, labels, threshold=0.5):
        return accuracy_score(labels, self.predict(X) > threshold)

    def eval_rocauc(self, X, labels):
        return roc_auc_score(labels, self.predict(X))

    def evaluate(self, X, labels):
        roc, auc = calc_scores(X.assign(nn_model=np.round(self.predict(X).flatten(), 0), label=labels), "nn_model") 
        return roc, auc
    @property
    def model_dict(self):
        return {"vectorizer": self.cv, "model": self.model}

    @classmethod
    def from_dict(cls, model_dict):
        "Get Model from dictionary"
        nbow_model = cls(len(model_dict["vectorizer"].vocabulary_))
        nbow_model.model = model_dict["model"]
        nbow_model.cv = model_dict["vectorizer"]
        return nbow_model

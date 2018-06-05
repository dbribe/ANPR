import os
import random
import tensorflow as tf
from ai_util import plot_images, extract_predictions
from config import MODEL_PATH
from model import build_model
from train import X_train, X2_train

g = tf.Graph()
with g.as_default():
    session = tf.InteractiveSession()
    model = build_model()
    saver = tf.train.Saver()
    saver.restore(session, os.path.join(MODEL_PATH, "model"))

    ids = [random.randint(0, X_train.shape[0] - 1) for _ in range(9)]
    predictions = model.output.eval(session=session, feed_dict={model.x_placeholder: X2_train[ids]})
    plot_images(X_train[ids], extract_predictions(predictions))

    session.close()

import tensorflow as tf
import random
import os
from config import MODEL_PATH
from ai_util import extract_predictions, plot_images
from model import build_model
from train import X2_test, X_test


g = tf.Graph()
with g.as_default():
    session = tf.InteractiveSession()
    model = build_model()
    saver = tf.train.Saver()
    saver.restore(session, os.path.join(MODEL_PATH, "model"))
    ids = [random.randint(0, X2_test.shape[0]-1) for _ in range(9)]
    predictions = model.output.eval(session=session, feed_dict={model.x_placeholder: X2_test[ids]})
    plot_images(X_test[ids], extract_predictions(predictions))
    session.close()

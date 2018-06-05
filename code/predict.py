import time
import tensorflow as tf
import numpy as np
import os
from model import build_model
from config import MODEL_PATH
from ai_util import extract_predictions, plot_images, load_image


start_time = time.time()
g = tf.Graph()
with g.as_default():
    session = tf.InteractiveSession()
    model = build_model()
    saver = tf.train.Saver()
    saver.restore(session, os.path.join(MODEL_PATH, "model"))
    image = np.array([load_image("poza4.jpg")])
    imagePredict = np.reshape(image, (image.shape[0], image.shape[1] * image.shape[2]))

    predictions = model.output.eval(session=session, feed_dict={model.x_placeholder: imagePredict})
    print(predictions[0][0])
    print(time.time() - start_time)
    plot_images(image, extract_predictions(predictions))

    session.close()
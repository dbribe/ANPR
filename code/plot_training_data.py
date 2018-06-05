from random import random
from train import X_train, Y_train
from ai_util import plot_images, extract_predictions


xs = [random.randint(0, X_train.shape[0] - 1) for _ in range(9)]
plot_images(X_train[xs], extract_predictions(Y_train[xs]))

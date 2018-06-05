from config import IMG_HEIGHT, IMG_WIDTH
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from pylab import rcParams
import os
from skimage import io
import json
import random
import numpy as np


def read_dir_files(dname):
    paths = []
    for file in os.listdir(os.path.join(dname, "img")):
        bname = os.path.basename(file).split(".")[0]
        img_name = os.path.join(dname, "img", file)
        ann_name = os.path.join(dname, "ann", bname + ".json")
        paths.append((img_name, ann_name))
    return paths


def read_paths(paths):
    all_paths = []
    for path in paths:
        temp_paths = read_dir_files(path)
        all_paths.extend(temp_paths)
    return all_paths


def load_data(paths):
    xs = []
    ys = []
    for ex_paths in paths:
        img_path = ex_paths[0]
        ann_path = ex_paths[1]
        xs.append(load_image(img_path))
        ys.append(load_annotation(ann_path))

    return np.array(xs), np.array(ys)


def load_image(fname):
    return io.imread(fname)[:,:] / 255.


def load_annotation(fname):
    with open(fname) as data_file:
        data = json.load(data_file)

    left = data["objects"][0]["points"]["exterior"][0][0]
    top = data["objects"][0]["points"]["exterior"][0][1]
    right = data["objects"][0]["points"]["exterior"][2][0]
    bottom = data["objects"][0]["points"]["exterior"][2][1]
    return normalize_bounding_box(left,top,right,bottom)


def normalize_bounding_box(xa, ya, xb, yb):
    center_x = xa + (xb - xa) / 2
    center_y = ya + (yb - ya) / 2
    width = xb - xa
    height = yb - ya

    cx = center_x / IMG_WIDTH
    cy = center_y / IMG_HEIGHT
    w = width / IMG_WIDTH
    h = height / IMG_HEIGHT

    label = np.array([cx, cy, w, h]) * 2 - (1, 1, 1, 1)
    return label


def extract_predictions(predictions):
    result = np.array([])
    first_time = True
    for prediction in predictions:
        center_x = prediction[0]
        center_y = prediction[1]
        width = prediction[2]
        height = prediction[3]
        center_x = (center_x + 1) / 2 * IMG_WIDTH
        center_y = (center_y + 1) / 2 * IMG_HEIGHT
        width = (width + 1) / 2 * IMG_WIDTH
        height = (height + 1) / 2 * IMG_HEIGHT
        xa = center_x - width / 2
        ya = center_y - height / 2
        xb = center_x + width / 2
        yb = center_y + height / 2
        sol = np.array([xa, ya, xb, yb])

        if first_time:
            first_time = False
            result = np.array([sol])
        else:
            result = np.append(result, [sol], axis=0)
    return result


def train_test_split(paths, train_percentage=.98):
    train_paths = []
    test_paths = []
    for path in paths:
        if random.random() < train_percentage:
            train_paths.append(path)
        else:
            test_paths.append(path)
    return train_paths, test_paths


def show_image(image, labels):
    rect = Rectangle((labels[0], labels[1]), labels[2] - labels[0], labels[3] - labels[1], edgecolor='r', fill=False)
    plt.imshow(image)
    gca = plt.gca()
    gca.add_patch(rect)


def plot_images(images, labels):
    rcParams['figure.figsize'] = 14, 8
    plt.gray()
    fig = plt.figure()
    for i in range(min(9, images.shape[0])):
        fig.add_subplot(3, 3, i + 1)
        show_image(images[i], labels[i])
    plt.show()


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
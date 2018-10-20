import numpy as np
from collections import Counter
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.models import load_model
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import itertools

def encode(labels):
    '''
    One-hot-encode labels.
    '''
    return to_categorical(labels)

def decode(labels):
    '''
    Decode one-hot-encoded labels.
    '''
    return np.argmax(labels, axis=1)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_metrics(history):
    '''
    Plots metrics returned by keras.CallBacks.History objects.
    Arguments:
        history: dictionary. Keys are metrics and entries are lists of values for each epoch.
    '''
    keys = list(history.keys())
    num_metrics = len(keys)
    fig, axs = plt.subplots(num_metrics, 1, figsize=(5, 5*num_metrics))
    fig.subplots_adjust(hspace=0.2)
    for i, key in enumerate(keys):
        y = history[key]
        x = range(1, len(y)+1)
        axs[i].plot(x, y)
        axs[i].set(ylabel=key)
        axs[i].set_xticks(x)
        yticks = np.arange(min(y), max(y), (max(y)-min(y))/10)
        axs[i].set_yticks(yticks)
    plt.xlabel('Epochs')

def plot_images(images, labels, predictions, classes=None, correct=True, incorrect=True):
    '''
    plot_images(images, labels, predictions, classes=None, correct=True, incorrect=True)
    Plots images with labels.
    Arguments:
        images: list of images.
        labels: list or array. True labels of images, must have length equal to 0 dimension of images.
        predictions: list or array. Predicted labels for images, must have length equal to 0 dimension of images.
        classes: list of str. Optional. Word labels for images.
        correct: bool. If True, images where true and predicted label match will be displayed.
        incorrect: bool. If True, images where true and predicted label do not match will be displayed.
    '''
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    corr_pred = [i for i,_ in enumerate(labels) if np.array_equal(labels[i], predictions[i])]
    length = 0
    cut = 0
    if correct:
        length += np.count_nonzero(corr_pred)
        cut += length
    if incorrect:
        length += np.size(corr_pred) - np.count_nonzero(corr_pred)
    fig, axs = plt.subplots(length, 1, figsize=(5, 5*length))
    correct_idx = np.where(corr_pred)[0]
    incorrect_idx = np.where(corr_pred == False)[0]

    for i, ax in enumerate(axs[:cut]):
        idx = correct_idx[i]
        if correct and corr_pred[idx]:
            title = 'Correctly predicted as '+ classes[labels[idx]] if classes else 'Correctly guessed'
            ax.imshow(images[idx])
            ax.set_title(title)
            ax.set_axis_off()

    for i, ax in enumerate(axs[cut:]):
        idx = incorrect_idx[i]
        if incorrect and (not corr_pred[idx]):
            title = classes[labels[idx]]+' but predicted '+classes[predictions[idx]] if classes else 'Incorrectly predicted'
            ax.imshow(images[idx])
            ax.set_title(title)
            ax.set_axis_off()

def compute_class_weights(train_gen):
    '''
    Returns class weights for imbalanced classes.
    Arguments:
        train_gen: batches generator from
                   keras.preprocessing.image.ImageDataGenerator().flow_from_directory()
    '''
    counter = Counter(train_gen.classes)                          
    max_val = float(max(counter.values()))       
    return {class_id : max_val/num_images for class_id, num_images in counter.items()}

def class_weights_from_labels(y_true):
    '''
    Returns class weights for imbalanced classes from ground-truth labels.
    '''
    count = np.sum(y_true, axis=0)
    max_value = max(count)
    norm = max_value/(count+1) #+1 to avoid division by zero
    k = np.empty((145))
    for i, val in enumerate(norm):
        if val < 10:
            k[i] = 1.
        elif val < 100:
            k[i] = 10.
        elif val < 1000:
            k[i] = 100.
        else:
            k[i] = 1000.
    return {i: k[i] for i in range(len(count))}

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def plot_f1(y_true, y_pred, average=None):
    preds = np.empty(y_pred.shape)
    f1_scores = []
    x = np.arange(0, 1.05, 0.05)
    iter_x = list(x)
    for threshold in iter_x:
        preds[y_pred>=threshold] = 1
        preds[y_pred<threshold] = 0
        f1_s = f1_score(y_true, preds, average=average)
        f1_scores.append(f1_s)
    plt.plot(x, f1_scores)
    plt.title('F1 Score vs. threshold')
    plt.xlabel('Threshold')
    plt.xticks(np.arange(0, 1, 0.1))
    plt.ylabel('F1 Score')
    plt.show()
    return iter_x, f1_scores
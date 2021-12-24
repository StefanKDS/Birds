from sklearn.preprocessing import LabelEncoder
from DatasetEngine import download_and_prepare_bird_dataset, get_concat_dataframe
from sklearn.model_selection import train_test_split
import pandas as pd
from helper_functions import plot_loss_curves, create_tensorboard_callback
import numpy as np
from Models import create_and_fit_dense_model, create_and_fit_cnn_model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# DOWNLOAD DATASETS
# IS ONLY NEEDED IF THERE'RE NO DATA PREVIOUSLY DOWNLOADED
#download_and_prepare_bird_dataset('northern%20cardinal', 5)
#download_and_prepare_bird_dataset('Gaviidae', 5)
#download_and_prepare_bird_dataset('Crypturellus%20cinereus', 5)

# GET DATAFRAMES
train_df = get_concat_dataframe(['northern%20cardinal', 'Gaviidae', 'Crypturellus%20cinereus'])

data = np.array(train_df['feature'].tolist())
labels = np.array(train_df['label'].tolist())

# ENCODE LABELS
labels_encoded = pd.get_dummies(labels)
encoder = LabelEncoder()
encoder.fit(labels)
np.save('Auswertung/classes.npy', encoder.classes_)

# SPLIT TRAINING AND VALIDATION DATA
X_train, X_test, y_train, y_test = train_test_split(data, labels_encoded, shuffle=True, test_size=0.3, random_state=42)

# CALLBACKS
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', min_lr=0.001, patience=10, mode='min', verbose=1)
early_stopping = EarlyStopping(patience=15, monitor='val_accuracy')
tensorboard = create_tensorboard_callback('Auswertung/', 'BirdSoundPrediction')
callbacks = [tensorboard]

# CREATE AND FIT MODEL
history = create_and_fit_dense_model(32, 148, callbacks, X_train, X_test, y_train, y_test)

# PLOT THE TRAINING CURVES
plot_loss_curves(history)
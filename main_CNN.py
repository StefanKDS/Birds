from sklearn.preprocessing import LabelEncoder
from DatasetEngine import download_and_prepare_bird_dataset, get_concat_dataframe
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D
import pandas as pd
from helper_functions import plot_loss_curves, create_tensorboard_callback
import numpy as np
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# DOWNLOAD DATASETS
# IS ONLY NEEDED IF THERE'RE NO DATA PREVIOUSLY DOWNLOADED
#download_and_prepare_bird_dataset('northern%20cardinal', 5)
#download_and_prepare_bird_dataset('Gaviidae', 5)
#download_and_prepare_bird_dataset('Crypturellus%20cinereus', 5)

# GET DATAFRAMES
train_df = get_concat_dataframe(['northern%20cardinal', 'Gaviidae', 'Crypturellus%20cinereus'])

# SHUFFLE TRAIN_DF
from sklearn.utils import shuffle
train_df_shuffeled = shuffle(train_df)

# CREATE LABELS AND DATA
data = np.array(train_df_shuffeled['feature'].tolist())
labels = np.array(train_df_shuffeled['label'].tolist())

#data = np.array(train_df['feature'].tolist())
#labels = np.array(train_df['label'].tolist())

labels_encoded = pd.get_dummies(labels)

encoder = LabelEncoder()
encoder.fit(labels)
np.save('Auswertung/classes.npy', encoder.classes_)

# SPLIT TRAINING AND VALIDATION DATA
X_train, X_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.3, random_state=42)

# CALLBACKS
#reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', min_lr=0.001, patience=5, mode='min', verbose=1)
#early_stopping = EarlyStopping(patience=10, monitor='val_accuracy')
tensorboard = create_tensorboard_callback('Auswertung/', 'BirdSoundPrediction')
callbacks = [tensorboard]

# BUILD THE MODEL
model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(40,1)))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(3, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# TRAIN THE MODEL
history_CNN = model.fit(X_train, y_train, batch_size=32, callbacks=callbacks, epochs=25, validation_data=(X_test, y_test))

# SAVE MODEL AND HISTORY_DATA
np.save('Auswertung/history_CNN.npy', history_CNN.history)
model.save('Auswertung/model_CNN')

# PLOT THE TRAINING CURVES
plot_loss_curves(history_CNN)
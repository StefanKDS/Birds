import os

from sklearn.preprocessing import LabelEncoder

from DatasetEngine import download_bird_dataset, get_concat_dataframe
from sklearn.model_selection import train_test_split
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.models import Sequential
from keras.layers import Dense, Dropout
import pandas as pd
from helper_functions import plot_loss_curves
import matplotlib.pyplot as plt
import numpy as np

# DOWNLOAD DATASETS
#download_bird_dataset('northern%20cardinal')
#download_bird_dataset('Gaviidae')

# GET DATAFRAMES
train_df = get_concat_dataframe(['northern%20cardinal', 'Gaviidae'])

# shuffle train_df
#from sklearn.utils import shuffle
#df = shuffle(train_df)

# CREATE LABELS AND DATA
labels = train_df['label']
data = train_df.drop(["label"], axis=1)

encoder = LabelEncoder()
encoder.fit(labels)
print(encoder.classes_)
np.save('Auswertung/classes.npy', encoder.classes_)

labels_encoded = encoder.transform(labels)

print(labels_encoded)

# PLOT FREQUENCY
#plt.plot(data[0], label="Frequency")
#plt.show()

# SPLIT TRAINING AND VALIDATION DATA
X_train, X_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.3, random_state=42)

model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(220500, 1)))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=2, epochs=2, validation_data=(X_test, y_test))

np.save('Auswertung/history.npy', history.history)
model.save('Auswertung/model')

# Plot the training curves
plot_loss_curves(history)
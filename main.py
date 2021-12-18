from DatasetDownloader import download_bird_dataset, get_concat_dataframe
from sklearn.model_selection import train_test_split
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.models import Sequential
from keras.layers import Dense, Dropout
import tensorflow as tf

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

print(labels[:10])
print(labels[:-10])

# PLOT FREQUENCY
import matplotlib.pyplot as plt
#plt.plot(data[0], label="Frequency")
#plt.show()
plt.plot(data[300], label="Frequency")
plt.show()
#plt.plot(data[150], label="Frequency")
#plt.show()

# SPLIT TRAINING AND VALIDATION DATA
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(220500, 1)))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=3, epochs=2, validation_data=(X_test, y_test))

# PREDICTION
from tensorflow import keras
from DatasetEngine import perpare_mp3_for_prediction
import numpy as np

# Load model
model = keras.models.load_model('Auswertung/model')
data = perpare_mp3_for_prediction('Data/Gaviidae/mp3/120507-002lomB.mp3')
#data = perpare_mp3_for_prediction('Data/northern_cardinal/mp3/CardinalKeyWest.mp3')

np_array=np.asarray(data)
reshaped_array = np_array.reshape( 220500, 1).T

print(data)

# Make a prediction
predArray = model.predict(reshaped_array)
pred = np.argmax(predArray, axis=1)

classes = np.load('Auswertung/classes.npy', allow_pickle=True)
print(classes[pred])
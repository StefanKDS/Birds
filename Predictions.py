# PREDICTION
from tensorflow import keras
from DatasetEngine import perpare_mp3_for_prediction, show_spectogram_for_mp3
import numpy as np

# Load model
model = keras.models.load_model('Auswertung/model')
#data = perpare_mp3_for_prediction('Data/Gaviidae/preTest/XC680885-GAVSTE2021-07-14-0644-TOLK-c-m.mp3')
#data = perpare_mp3_for_prediction('Data/northern_cardinal/mp3/CardinalKeyWest.mp3')
data = perpare_mp3_for_prediction('Data/Crypturellus_cinereus/mp3/Cinereous_tinamou1.mp3')

# Make a prediction
predArray = model.predict(data)
pred = np.argmax(predArray, axis=1)

print(pred)

classes = np.load('Auswertung/classes.npy', allow_pickle=True)
print(classes[pred])

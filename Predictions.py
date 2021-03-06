# PREDICTION
from tensorflow import keras
from DatasetEngine import perpare_mp3_for_prediction, show_spectogram_for_mp3
import numpy as np

# Load model
#model = keras.models.load_model('Auswertung/model')
model_CNN = keras.models.load_model('Auswertung/model_CNN')

#data = perpare_mp3_for_prediction('Data/Gaviidae/preTest/XC680885-GAVSTE2021-07-14-0644-TOLK-c-m.mp3')
#data = perpare_mp3_for_prediction('Data/northern_cardinal/preTest/XC677300-NOCA_SGC_Sep_28_2021.mp3')
#data = perpare_mp3_for_prediction('Data/Crypturellus_cinereus/mp3/Cinereous_tinamou1.mp3')
#data = perpare_mp3_for_prediction('Data/Cisticola_juncidis/preTest/XC733052-20220617_0712_cisticole.mp3')

# Make a prediction
predArray = model_CNN.predict(data)
pred = np.argmax(predArray, axis=1)

print(pred)

classes = np.load('Auswertung/classes.npy', allow_pickle=True)
print(classes[pred])

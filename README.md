# Birds

This ML model learns to differentiate between bird species by means of mp3 recordings of the voices.
The dataset are downloded from the website https://xeno-canto.org/.

# What is what ?

###<ins>main.py</ins>
The complete script to read the data, prepare it, create and fit the model.

###<ins>DatasetEngine.py</ins>
Load and prepare the Dataset

###<ins>helper_functions.py</ins>
Some small helpers

###<ins>Models.py</ins>
Here you can find the different models like CNN or Dense model

###<ins>Predictions.py</ins>
Make predictions with the prepared models

###<ins>TensorboardView.py</ins>
Show the histories in Tensorboard

###<ins>TextBox.py</ins>
Place for experiments


# How to train the model

First you have to get and prepare the datasets. This can be done with the function 'download_bird_dataset'.
As parameter you give him the name of the species in query form of the xeno API.

E.g.:  
download_and_prepare_bird_dataset('northern%20cardinal', 5)  
download_and_prepare_bird_dataset('Gaviidae', 5)

This can take a while...

Here you can see the way from the wavefile to the extracted features:

![Alt text](Doc/Diagrams.jpg?raw=true)

This brings the data down to 1kb and garantues a fast model !

Now you can get the trainingdata with the function 'get_concat_dataframe'. As parameter you give him a list of the 
species you want him to learn.

E.g.:  
train_df = get_concat_dataframe(['northern%20cardinal', 'Gaviidae'])


And then the magic starts.......

Here are the accuracy diagrams:

![Alt text](Doc/Diagrams2.jpg?raw=true)


# How to make predictions

Man, it's so easy....

from tensorflow import keras  
from DatasetEngine import perpare_mp3_for_prediction  

model = keras.models.load_model('Auswertung/model')  
data = perpare_mp3_for_prediction('Data/Gaviidae/mp3/120507-002lomB.mp3')  

pred = model.predict(data)  

# That's all folks

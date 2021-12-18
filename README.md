# Birds

This ML model learns to differentiate between bird species by means of mp3 recordings of the voices.
The dataset are downloded from the website https://xeno-canto.org/.

# How to train the model

First you have to get and prepare the datasets. This can be done with the function 'download_bird_dataset'.
As parameter you give him the name of the species in query form of the xeno API.

E.g.:
download_bird_dataset('northern%20cardinal')
download_bird_dataset('Gaviidae')

This can take a while...

Now you can get the trainingdata with the function 'get_concat_dataframe'. As parameter you give him a list of the 
species you want him to learn.

E.g.:
train_df = get_concat_dataframe(['northern%20cardinal', 'Gaviidae'])


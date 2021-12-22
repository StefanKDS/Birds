import wget
import json
import glob
import numpy as np
from pydub import AudioSegment
import librosa
import os
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import librosa.display


def downloas_bird_sounds(query):
    # GET ALL FILENAMES AND PATHS OF THE QUERY BIRD
    url = 'https://www.xeno-canto.org/api/2/recordings?query=' + query

    dataFolder = "Data/" + query.replace("%20", "_") + "/"
    mp3Folder = dataFolder + "mp3/"
    arrayFolder = dataFolder + "arrays/"
    predTestFolder = dataFolder + "preTest/"

    os.mkdir(dataFolder)
    os.mkdir(mp3Folder)
    os.mkdir(arrayFolder)
    os.mkdir(predTestFolder)

    filename = wget.download(url, dataFolder + 'recordings.json')
    print(filename)

    # Get the json entries from your downloaded json
    jsonFile = open(dataFolder + 'recordings.json', 'r')
    values = json.load(jsonFile)
    jsonFile.close()

    # Create a pandas dataframe of records & convert to .csv file
    record_df = pd.DataFrame(values['recordings'])
    record_df.to_csv(dataFolder + 'xc-noca.csv', index=False)

    # Make wget input file
    url_list = []
    for file in record_df['file'].tolist():
        url_list.append('{}'.format(file))
    with open(dataFolder + 'xc-noca-urls.txt', 'w+') as f:
        for item in url_list:
            f.write("{}\n".format(item))

    # Get all soundfiles
    os.system('wget -P ' + mp3Folder + ' --trust-server-names -i' + dataFolder + 'xc-noca-urls.txt')

    files = os.listdir(mp3Folder)
    [os.replace(mp3Folder + file, mp3Folder + file.replace(" ", "_")) for file in files]


def prepare_dataset(query, nbrOfTestSoundsForPrediction):
    dataFolder = "Data/" + query.replace("%20", "_") + "/"
    mp3Folder = dataFolder + "mp3/"
    arrayFolder = dataFolder + "arrays/"
    predTestFolder = dataFolder + "preTest/"

    # The following line is only needed once if ffmpeg is not part of the PATH variables
    # os.environ["PATH"] += os.pathsep + r'F:\ffmpeg\bin'

    # Reformat path string
    globlist = glob.glob(mp3Folder + "*.mp3")
    new_list = []
    for string in globlist:
        new_string = string.replace("\\", "/")
        new_list.append(new_string)

    # Copy 5 entries to /predTest
    last_elements = new_list[-nbrOfTestSoundsForPrediction:]
    print(last_elements)

    for file in last_elements:
        shutil.copy(file, predTestFolder)
        os.remove(file)

    globlist.clear()
    globlist = glob.glob(mp3Folder + "*.mp3")
    new_list.clear()
    for string in globlist:
        new_string = string.replace("\\", "/")
        new_list.append(new_string)

    # Extract frequencies and save them as np array
    for file in new_list:
        src = file
        dst = "tmp/tmp.wav"

        # convert mp3 to wav
        sound = AudioSegment.from_mp3(src)
        ten_seconds = 10 * 1000
        first_10_seconds = sound[:ten_seconds]
        first_10_seconds.export(dst, format="wav")

        y, sr = librosa.load(dst)
        librosa.util.fix_length(y, 220500)

        mfccs_features = librosa.feature.mfcc(y, sr, n_mfcc=40)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

        index = new_list.index(file)
        arrayPath = arrayFolder + str(index)

        np.save(arrayPath, mfccs_scaled_features)

        # Remove temp wav file
        os.remove(dst)

    arraylist = glob.glob(arrayFolder + "*.npy")

    extracted_features = []
    for file in arraylist:
        data = np.load(file)
        label = query.replace("%20", "_")
        extracted_features.append([data, label])

    np.save(arrayFolder + "summery_array", extracted_features)


def download_and_prepare_bird_dataset(query, nbrOfTestSoundsForPrediction):
    downloas_bird_sounds(query)
    prepare_dataset(query, nbrOfTestSoundsForPrediction)


def show_spectogram_for_mp3(filepath):
    src = filepath
    dst = "tmp/tmp.wav"

    # convert wav to mp3
    sound = AudioSegment.from_mp3(src)
    ten_seconds = 10 * 1000
    first_10_seconds = sound[:ten_seconds]
    first_10_seconds.export(dst, format="wav")

    y, sr = librosa.load(dst)
    y_norm = librosa.util.normalize(y)

    # x-axis has been converted to time using our sample rate.
    # matplotlib plt.plot(y), would output the same figure, but with sample
    # number on the x-axis instead of seconds
    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(y_norm, sr=sr)
    plt.show

    S = librosa.feature.melspectrogram(y=y_norm, sr=sr)

    print(S.shape)

    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr,fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.show()


def perpare_mp3_for_prediction(filepath):
    src = filepath
    dst = "tmp/tmp2.wav"

    # convert mp3 to wav
    sound = AudioSegment.from_mp3(src)
    ten_seconds = 10 * 1000
    first_10_seconds = sound[:ten_seconds]
    first_10_seconds.export(dst, format="wav")

    y, sr = librosa.load(dst)
    librosa.util.fix_length(y, 220500)

    mfccs_features = librosa.feature.mfcc(y, sr, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

    mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)

    # Remove temp wav file
    os.remove(dst)

    return mfccs_scaled_features


def get_dataframe(query):
    dataFolder = "Data/" + query.replace("%20", "_") + "/"
    arrayFolder = dataFolder + "arrays/"

    numpy_data = np.load(arrayFolder + "summery_array.npy", allow_pickle=True)
    extracted_features_df = pd.DataFrame(numpy_data, columns=['feature', 'label'])

    return extracted_features_df


def get_concat_dataframe(query_list):
    result = pd.DataFrame()

    for query in query_list:
        df = get_dataframe(query)
        result = result.append(df)

    return result

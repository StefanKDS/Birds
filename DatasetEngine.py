import wget
import json
import pandas as pd
import glob
import numpy as np
from pydub import AudioSegment
import librosa
import os
import pandas as pd


def download_bird_dataset(query):
    # GET ALL FILENAMES AND PATHS OF THE QUERY BIRD
    url = 'https://www.xeno-canto.org/api/2/recordings?query=' + query

    dataFolder = "Data/" + query.replace("%20", "_") + "/"
    mp3Folder = dataFolder + "mp3/"
    arrayFolder = dataFolder + "arrays/"

    os.mkdir(dataFolder)
    os.mkdir(mp3Folder)
    os.mkdir(arrayFolder)

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

    # --------------------------------------------------------------------------------------------------

    # Extract the frequencies and save them as NPArray
    files = os.listdir(mp3Folder)
    print(files)
    [os.replace(mp3Folder + file, mp3Folder + file.replace(" ", "_")) for file in files]

    # The following line is only needed once if ffmpeg is not part of the PATH variables
    # os.environ["PATH"] += os.pathsep + r'F:\ffmpeg\bin'

    # Reformat path string
    globlist = glob.glob(mp3Folder + "*.mp3")
    new_list = []
    for string in globlist:
        new_string = string.replace("\\", "/")
        new_list.append(new_string)

    nbr_of_entries = len(new_list)

    # Extract frequencies and save them as np array
    for file in new_list:
        src = file
        dst = "tmp/tmp.wav"

        # convert wav to mp3
        sound = AudioSegment.from_mp3(src)
        ten_seconds = 10 * 1000
        first_10_seconds = sound[:ten_seconds]
        first_10_seconds.export(dst, format="wav")

        y, sr = librosa.load(dst)
        y_norm = librosa.util.normalize(y)

        addeditems = 220500 - len(y_norm)
        if addeditems > 0:
            y_norm.resize((220500,), refcheck=False)
            for i in range(addeditems):
                y_norm[:-i] = 0.

        y_norm.resize((220500,), refcheck=False)

        index = new_list.index(file)
        arrayPath = arrayFolder + str(index)

        np.save(arrayPath, y_norm)

        # Remove temp wav file
        os.remove(dst)

    arraylist = glob.glob(arrayFolder + "*.npy")
    summery_array = np.zeros((495, 220500))

    for file in arraylist:
        freq_array = np.load(file)
        file_index = arraylist.index(file)
        summery_array[file_index - 1] = freq_array;

    np.save(arrayFolder + "summery_array", summery_array)


def perpare_mp3_for_prediction(filepath):
    src = filepath
    dst = "tmp/tmp2.wav"

    # convert wav to mp3
    sound = AudioSegment.from_mp3(src)
    ten_seconds = 10 * 1000
    first_10_seconds = sound[:ten_seconds]
    first_10_seconds.export(dst, format="wav")

    y, sr = librosa.load(dst)
    y_norm = librosa.util.normalize(y)

    addeditems = 220500 - len(y_norm)
    if addeditems > 0:
        y_norm.resize((220500,), refcheck=False)
        for i in range(addeditems):
            y_norm[:-i] = 0.

    y_norm.resize((220500,), refcheck=False)

    # Remove temp wav file
    os.remove(dst)

    print(y_norm.shape)

    return y_norm


def get_dataframe(query):
    import pandas as pd

    dataFolder = "Data/" + query.replace("%20", "_") + "/"
    arrayFolder = dataFolder + "arrays/"

    numpy_data = np.load(arrayFolder + "summery_array.npy")

    df = pd.DataFrame(numpy_data)
    df['label'] = query.replace("%20", "_")

    return df


def get_concat_dataframe(query_list):
    result = pd.DataFrame()

    for query in query_list:
        df = get_dataframe(query)
        result = result.append(df)

    return result

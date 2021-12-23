from helper_functions import launchTensorBoard
import threading

t = threading.Thread(target=launchTensorBoard('Auswertung/BirdSoundPrediction/20211223-093239'), args=([]))
t.start()
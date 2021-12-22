from helper_functions import launchTensorBoard
import threading

t = threading.Thread(target=launchTensorBoard('Auswertung/BirdSoundPrediction/20211222-155431'), args=([]))
t.start()
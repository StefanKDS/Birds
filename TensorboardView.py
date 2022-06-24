from helper_functions import launchTensorBoard
import threading

t = threading.Thread(target=launchTensorBoard('Auswertung/BirdSoundPrediction/20220624-124918'), args=([]))
t.start()
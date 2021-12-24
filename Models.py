import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D


def create_and_fit_dense_model(batch_size, epochs, callbacks, X_train, X_test, y_train, y_test):
    # BUILD THE MODEL
    model = Sequential()
    model.add(Dense(100, input_shape=(40,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    # TRAIN THE MODEL
    history = model.fit(X_train, y_train, batch_size=batch_size, callbacks=callbacks, epochs=epochs,
                        validation_data=(X_test, y_test))

    # SAVE MODEL AND HISTORY_DATA
    np.save('Auswertung/history.npy', history.history)
    model.save('Auswertung/model')

    return history


def create_and_fit_cnn_model(batch_size, epochs, callbacks, X_train, X_test, y_train, y_test):
    # BUILD THE MODEL
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(40, 1)))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    # TRAIN THE MODEL
    history_CNN = model.fit(X_train, y_train, batch_size=32, callbacks=callbacks, epochs=25,
                            validation_data=(X_test, y_test))

    # SAVE MODEL AND HISTORY_DATA
    np.save('Auswertung/history_CNN.npy', history_CNN.history)
    model.save('Auswertung/model_CNN')

    return history_CNN
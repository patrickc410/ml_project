import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.utils import to_categorical
import tensorflow_addons as tfa



class TFMusicClassifier():
    def __init__(self, DIM_IN:int=11, DIM_HIDDEN:int=16,  DIM_OUT:int=13) -> None:
        self.nn = Sequential()
        self.nn.add(Dense(DIM_HIDDEN, input_shape=(DIM_IN,), activation='relu'))
        self.nn.add(Dropout(0.1))
        self.nn.add(Dense(DIM_HIDDEN, activation='relu'))
        self.nn.add(Dropout(0.1))
        self.nn.add(Dense(DIM_OUT, activation='relu'))
        self.nn.add(Activation("softmax"))

        self.nn.compile(
            # loss='categorical_crossentropy',
            loss=tfa.losses.SigmoidFocalCrossEntropy()
            optimizer='adam',
            metrics=['accuracy']
        )
            
        self.DIM_IN = DIM_IN
        self.DIM_HIDDEN = DIM_HIDDEN
        self.DIM_OUT = DIM_OUT
        
    def summary(self):
        return self.nn.summary()
    
    def fit(self, X_train, y_train, X_test, y_test, epochs:int=20, batch_size:int=16, lr:float=0.005):
        y_train_cat = to_categorical(y_train, self.DIM_OUT)
        y_test_cat = to_categorical(y_test, self.DIM_OUT)
        print(f"Train categorical labels shape: {y_train_cat.shape}")
        print(f"Test categorical labels shape:  {y_test_cat.shape}")

        history = self.nn.fit(
            X_train, y_train_cat,
            batch_size=batch_size,
            lr=lr,
            epochs=epochs,
            shuffle=True,
            verbose=1,
            initial_epoch=0, 
            validation_data=(X_test, y_test_cat))
        return history
    
    def evaluate(self, X_test, y_test, ):
        y_test_cat = to_categorical(y_test, self.DIM_OUT)
        score = self.nn.evaluate(X_test, y_test_cat, verbose=0)
        return score
    

    


    
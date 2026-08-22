from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

def ann_model():
    """this function builds ann model"""

    model = Sequential([
        Input(shape=(4,)),
        Dense(8, activation='relu'),
        Dense(12, activation='relu'),
        Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def dnn_model():
    """this function builds dnn model"""

    model = Sequential([
        Input(shape=(4,)),
        Dense(16, activation='relu'),
        Dense(32, activation='relu'),
        Dense(20, activation='relu'),
        Dense(3, activation="softmax")
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

if __name__ == "__main__":
    ann = ann_model()
    dnn = dnn_model()

    print("ANN created successfully!")
    ann.summary()

    print("DNN created successfully!")
    dnn.summary()
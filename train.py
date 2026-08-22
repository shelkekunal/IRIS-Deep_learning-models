from data import load_data, data_preprocessing
from model import ann_model, dnn_model

x, y = load_data()
print('data loading successfully')

X_train_scaled, X_test_scaled, y_train, y_test = data_preprocessing(x, y)
print('data preprocessed successful')


ann = ann_model()
print("building ann model")


ann_histroy = ann.fit(X_train_scaled, y_train, epochs=50, 
                      validation_split=0.2, batch_size=10)
print('fit ann model successfully')

ann.save("models/ann.keras")
print('ann model save successfully')

print('building dnn model')
dnn = dnn_model()

dnn_history = dnn.fit(X_train_scaled, y_train, epochs=50, 
                      validation_split=0.2, batch_size=10)
print('fit dnn model successfully')


dnn.save("models/dnn.keras")
print('dnn model save successfully')

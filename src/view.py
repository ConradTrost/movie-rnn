import matplotlib.pyplot as plt
from tensorflow import keras
import joblib

model = keras.models.load_model('../_models/movie')
history = joblib.load('../_models/history.json')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

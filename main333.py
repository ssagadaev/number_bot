import autokeras as ak
from tensorflow.keras.datasets import mnist

# Загрузка данных MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255

# Создание модели для классификации изображений
clf = ak.ImageClassifier(max_trials=10) # max_trials - количество попыток для поиска лучшей модели

# Обучение модели
clf.fit(x_train, y_train, epochs=10)

# Оценка модели
_, accuracy = clf.evaluate(x_test, y_test)
print('Accuracy: %.2f' % (accuracy * 100))

# Сохранение модели
model = clf.export_model()
model.save('autokeras_mnist_model.h5')
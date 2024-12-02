import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras import layers

def build_and_train_model(model, x_train, y_train, x_test, y_test, epochs=5):
    # Compilăm modelul
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Antrenăm modelul
    model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))

    # Evaluăm modelul
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('\nAcuratețea pe setul de date de testare:', test_acc)

def main():
    # Încărcăm setul de date MNIST
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    print("Datele au fost încărcate cu succes!")

    # Preprocesăm datele
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Model 1: Modelul de bază
    print("\nModel 1: Modelul de bază")
    model_1 = keras.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    build_and_train_model(model_1, x_train, y_train, x_test, y_test)

    # Model 2: Mai mulți neuroni și un strat suplimentar
    print("\nModel 2: Mai mulți neuroni și un strat suplimentar")
    model_2 = keras.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    build_and_train_model(model_2, x_train, y_train, x_test, y_test)

    # Model 3: Mai mulți neuroni și mai multe straturi
    print("\nModel 3: Mai mulți neuroni și mai multe straturi")
    model_3 = keras.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    build_and_train_model(model_3, x_train, y_train, x_test, y_test)

    # Model 4: Mai puțini neuroni și mai multe straturi
    print("\nModel 4: Mai puțini neuroni și mai multe straturi")
    model_4 = keras.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    build_and_train_model(model_4, x_train, y_train, x_test, y_test)

if __name__ == "__main__":
    main()

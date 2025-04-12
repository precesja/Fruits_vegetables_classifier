import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

training_path = "dataset/Training"
test_path = "dataset/Test"

def image_preparation(path, fruits_names):
    x, y = [], []

    for label_index, label_name in enumerate(fruits_names):
        fruits_path = os.path.join(path, label_name)
        for fruit_name in os.listdir(fruits_path):
            fruit_path = os.path.join(fruits_path, fruit_name)
            try:
                img = Image.open(fruit_path).resize((100, 100))
                img_array = np.array(img) / 255.0
                if img_array.shape == (100, 100, 3):
                    x.append(img_array)
                    y.append(label_index)
            except Exception as e:
                print(f"Something went wrong: {e} ")
    return np.array(x, dtype=np.float32), to_categorical(y)
def build_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def main():

    fruits_names = os.listdir(training_path)

    X_train, Y_train = image_preparation(training_path, fruits_names)
    X_test, Y_test = image_preparation(test_path, fruits_names)

    model = build_model(len(fruits_names))

    history = model.fit(X_train, Y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(X_test, Y_test))

    model.save("fruit_classifier_model.keras")

    # Accuracy
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Loss
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
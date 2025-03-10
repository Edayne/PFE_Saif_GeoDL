#Import des librairies
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # type: ignore
from sklearn.model_selection import train_test_split
import os
import cv2
import time
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Create the dataset
def images_from_folder(folder):
    """Fonction générant un dataset depuis un dossier d'images de manière récursive.

    Args:
        folder (string): chemin vers le dossier contenant les images.

    Returns:
        list: Liste d'images à analyser.
    """
    images = []
    filenames = []
    for root, _, files in os.walk(folder):  # Parcours récursif du dossier Image_data
        for filename in files:
            
            if filename.endswith("Fe.tif") \
            or filename.endswith("Ca.tif") \
            or filename.endswith("Si.tif") \
            or filename.endswith("Al.tif") \
            or filename.endswith("Na.tif") :
                
                if filename.startswith("SL7D") \
                or filename.startswith("SL7A") \
                or filename.startswith("SL7B") \
                or filename.startswith("SL7bis") \
                or filename.startswith("SLG3233") : # On évite celles-ci car elles posent probleme
                    continue
                imagePath = os.path.join(root, filename)
                img = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
                img = cv2.resize(img, (512, 512))
                
                if img is not None: 
                    img = img / 255.0  
                    # Expand dimensions if grayscale (single channel)
                    if len(img.shape) == 2:  # If image is (H, W), make it (H, W, 1)
                        img = np.expand_dims(img, axis=-1)
                    images.append(img)
                    filenames.append(filename)
                    
    images = np.array(images, dtype=np.float32)
    return images, filenames

# 2. Build the CNN model

def data_augmentor(X_train):
    """Data Augmentation

    Args:
        X_train (np.ndarray): Jeu d'entrainement

    Returns:
        np.ndarray: Jeu de donnée augmenté
    """
    # data_augmentation = keras.Sequential([
    #     layers.RandomFlip("horizontal_and_vertical"),  
    #     layers.RandomRotation(0.15),  
    #     layers.RandomZoom(0.10),  
    #     layers.RandomContrast(0.1),  
    # ])
    data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),   # Increased rotation range
    layers.RandomZoom(0.2),       # Increased zoom range
    layers.RandomContrast(0.2),   # More contrast variation
    layers.RandomTranslation(0.1, 0.1)  # Random shift
])

    return data_augmentation(X_train)

def build_model(input_shape, num_classes):
    """Création du CNN

    Args:
        input_shape (tuple): _description_. 
        num_classes (int): Nombre de classes à décrire. 
        Doit correspondre à nombre de minéraux étudiés + 3 .

    Returns:
        keras.model: Notre modèle compilé
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # Convolutional layers

        layers.Conv2D(16, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.MaxPooling2D((2, 2), strides=2),

        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        # layers.Dropout(0.4),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.MaxPooling2D((2, 2), strides=2),

        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.MaxPooling2D((2, 2), strides=2),

        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.Dropout(0.3),

        layers.GlobalAveragePooling2D(), # Au lieu de Flatten()
        layers.Dense(64,
                     ), # kernel_regularizer=keras.regularizers.l2(0.001)
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.Dropout(0.5),

        layers.Dense(num_classes,           # One neuron per class
                     activation='sigmoid')  # Multi-label output
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# 3. Train the model
def train_model(model, X_train, y_train, X_val, y_val, epochs=20,):
    """Entraine le modèle fourni

    Args:
        model (keras.model): Modèle compilé
        X_train (numpy.ndarray): Images d'entrainement
        y_train (numpy.ndarray): Labels des images d'entrainement
        X_val (numpy.ndarray): Images de validation
        y_val (numpy.ndarray): Labels des images de validation
        epochs (int, optional): Nombre d'époques. Defaults to 20.

    Returns:
        History:  A History object. Its History.history attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable). 
    """
    # Création des callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, start_from_epoch=6)
    model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    
    # Entrainement du modèle
    print("\nDébut entrainement...")
    history = model.fit(
        X_train, y_train,
        validation_data = (X_val, y_val),
        epochs = epochs,
        batch_size = max(1, len(X_train)//10),
        callbacks = [early_stopping, model_checkpoint, reduce_lr],
        verbose = 1
    )
    return history
    
# 4. Evaluate the model
def evaluate_model(model, x_test, y_test):
    """Evalue les performances du modèle

    Args:
        model (keras.model): _description_
        x_test (_type_): _description_
        y_test (_type_): _description_

    Returns:
        int: 0
    """
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test loss: {test_loss:.2f}")
    print(f"Test accuracy: {test_acc:.2f}\n")
    return 0

def graph_plot_save(history):
    """Plots out the accuracy and loss of the model during training

    Args:
        history (History): History object given after keras.model.fit
    """

    #plotting Accuracy
    plt.subplot(121)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Précision du modèle')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    #plotting Loss
    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Perte du modèle')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    results_dir = Path("Results")
    results_dir.mkdir(parents=True, exist_ok=True)
    # Find the next available index
    existing_files = [f.stem for f in results_dir.glob("results_*.png")]
    existing_numbers = [int(f.split("_")[-1]) for f in existing_files if f.split("_")[-1].isdigit()]
    next_index = max(existing_numbers, default=0) + 1  # Default to 1 if no files exist
    # Save the plot with an incremented filename
    plot_path = results_dir / f"results_{next_index}.png"
    plt.savefig(plot_path)

    print(f"Plot saved as: {plot_path}")

    plt.show()
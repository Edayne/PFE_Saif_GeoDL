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
            or filename.endswith("Si.tif") :
                
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
def build_model(input_shape, num_classes):
    """Création du CNN

    Args:
        input_shape (tuple, optional): _description_. 
        num_classes (int, optional): _description_. Defaults to 3.

    Returns:
        _type_: _description_
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        # Couche d'augmentation
        layers.RandomFlip("horizontal"),  # Randomly flip images horizontally
        layers.RandomRotation(0.05),  # Rotate up to ±10%
        layers.RandomZoom(0.2),  # Random zoom in/out
        layers.RandomContrast(0.1),  # Slight contrast variations

        # Couches de convolution
        layers.Conv2D(32, (3, 3), 
                      activation='relu', 
                      padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), 
                      activation='relu', 
                      padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), 
                      activation='relu', 
                      padding='same'),
        layers.MaxPooling2D((2, 2)),
        # layers.Conv2D(256, (3, 3), 
        #               activation='relu', 
        #               padding='same'),
        
        layers.Flatten(),
        
        # layers.Dense(256, 
        #              activation='relu'),
        # layers.Dropout(0.50),
        
        layers.Dense(128, 
                     activation='relu'),
        layers.Dropout(0.50),
        layers.BatchNormalization(),
        layers.Dense(num_classes, # Un neurone par classe
                     activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# 3. Train the model
def train_model(model, X_train, y_train, X_test, y_test, epochs=15):
    """_summary_

    Args:
        model (keras.model): Le modèle que l'on souhaite entrainer
        X_train (_type_): _description_
        y_train (_type_): _description_
        X_test (_type_): _description_
        y_test (_type_): _description_
        epochs (int, optional): _description_. Defaults to 10.
        batch_size (int, optional): _description_. Defaults to 64.

    Returns:
        keras.model: Modèle entrainé
    """
    # Création des callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    # Entrainement du modèle
    print("\nDébut entrainement...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=max(1, len(X_train)//10),
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        verbose=1
    )
    return history
    
# 4. Evaluate the model
def evaluate_model(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}\n")


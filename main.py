from model_building import *
from y_label_generator import *

import matplotlib.pyplot as plt


if __name__ == "__main__":
    print(f"\nTensorflow version : {tf.__version__}") #2.18.0
    print(f"Keras version : {tf.keras.__version__}") #3.8.0
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
    # Import des images
    print("\nImport des images et création du dataset...")
    
    images_path = "./Image_data"
    csv_path = "./sample_data.csv"
    
    images, filenames = images_from_folder(images_path)
    print(f"\nNombre d'images importées : {len(images)}")
    df_labels, encoded_labels, encoder = generate_labels(images_path, csv_path)
    
    df_labels.to_csv("df_labels.csv")
    print(df_labels)
    
    y = encoded_labels
    
    
    # Séparation du dataset en Train/Test
    X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(images, y, filenames, test_size=0.10)
    print("Nombre d'images dans le Training set: ",len(X_train))
    print("Nombre d'images dans le Test set: ",len(X_test))
    img_shape = X_train[0].shape
    
    # Construction du modèle
    MODEL_PATH = "best_model.keras"
    EPOCHS = 25
    NB_CLASS = encoded_labels.shape[1]
    
    print("\nDébut entrainement...")

    model = build_model(input_shape=img_shape, num_classes=NB_CLASS)
    model.summary()

    # Entrainement du modèle
    start = time.time()
    history = train_model(model, X_train, y_train, X_test, y_test, epochs=EPOCHS)
    end = time.time()
    print(f"Temps d'éxécution: {end-start:.2f}s")
    
    evaluate_model(model, X_test, y_test)
    
    # Affichage des prédictions 
    y_test_pred = model.predict(X_test)
    y_pred_bin = (y_test_pred > 0.5).astype(int)
    label_names = encoder.get_feature_names_out()
    all_predicted_labels = []
    for i, sample in enumerate(y_pred_bin):
        predicted_labels = [label_names[j] for j in range(len(sample)) if sample[j] == 1]
        all_predicted_labels.append(predicted_labels)
    
    for filename, label in zip(filenames_test, all_predicted_labels):    
        print(f"{filename}: prédiction {label}")
        

    # On crée un dataframe associant chaque nom de fichier à sa prédiction
    # On le stocke dans un CSV
    df_results = pd.DataFrame({
        "Filename": filenames_test,
        "Predicted Label": all_predicted_labels
    })

    df_results.to_csv("predictions.csv", index=False)

    print("Predictions saved to predictions.csv!")
    
    #plotting Accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Précision modèle')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('Accuracy.png')
    plt.show()
    plt.savefig('Loss and validation loss plot.png')
    print("ACCURACY GRAPH SAVED !")
    
    #plotting Loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Perte du modèle')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('Loss and validation loss plot.png')
    plt.show()
    plt.savefig('loss.png')
    print("LOSS GRAPH SAVED !")
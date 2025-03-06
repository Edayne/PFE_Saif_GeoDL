from model_building import *
from y_label_generator import *

import matplotlib.pyplot as plt
import sys
from pathlib import Path



if __name__ == "__main__":
    print(f"\nTensorflow version : {tf.__version__}") #2.18.0
    print(f"Keras version : {tf.keras.__version__}") #3.8.0
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
    # Import des images
    print("\nImport des images et création du dataset...")
    
    images_path = "./Image_data"
    csv_path = "./sample_data.csv"

    if not os.path.exists(images_path):
        print("IMAGE FOLDER NOT FOUND !")
        sys.exit(-1)


    images, filenames = images_from_folder(images_path)
    print(f"\nNombre d'images importées : {len(images)}")
    df_labels, encoded_labels, encoder = generate_labels(images_path, csv_path)
    
    df_labels.to_csv("df_labels.csv")
    print(df_labels)
    
    y = encoded_labels
    
    
    # Séparation du dataset en Train/Test
    X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(images, y, filenames, test_size=0.20)
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
    plt.subplot(121)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Précision du modèle')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    
    #plotting Loss
    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Perte du modèle')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

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
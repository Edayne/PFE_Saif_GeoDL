from model_building import *
from y_label_generator import *

import sys
from sklearn.utils import class_weight


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

    # Séparation du dataset en Test/Val puis Val séparé en Val/Test
    # Random seed = 123 ; à supprimer pour avoir une meilleure idée de la qualité du modèle
    X_train, X_val, y_train, y_val, filenames_train, filenames_val \
        = train_test_split(images, y, filenames, 
                           test_size=0.20, 
                           shuffle=True, stratify=y, 
                           random_state=123)
    
    X_train = data_augmentor(X_train)

    X_test, X_val, y_test, y_val, filenames_test, filenames_val \
        = train_test_split(X_val, y_val, filenames_val, 
                           test_size=0.5,
                           random_state=123)

    print("Nombre d'images dans le Training set: ",len(X_train))
    print("Nombre d'images dans le Validation set: ",len(X_val))
    print("Nombre d'images dans le Test set: ",len(X_test))

    # Construction du modèle
    img_shape = X_train[0].shape
    MODEL_PATH = "best_model.keras"
    EPOCHS = 25
    NB_CLASS = encoded_labels.shape[1]

    if os.path.exists(MODEL_PATH):
        choice = input("\nUn modèle déjà entrainé a été trouvé. \nVoulez-vous en en(T)rainer un nouveau ou (E)valuer celui existant? [t/e]: ").strip().lower()

        if choice == 'e':
            print("Chargement...")
            model = tf.keras.models.load_model(MODEL_PATH)
            
            evaluate_model(model, X_test, y_test)

        elif choice == 't':
            model = build_model(input_shape=img_shape, num_classes=NB_CLASS)
            model.summary()

            # Entrainement du modèle
            start = time.time()
            history = train_model(model, X_train, y_train, X_val, y_val, epochs=EPOCHS)
            end = time.time()
            print(f"Temps d'éxécution: {end-start:.2f}s\n")
            
            evaluate_model(model, X_test, y_test)
            graph_plot_save(history)

        else:
            print("INVALIDE. SORTIE.")
    else:
        print("Aucun modèle trouvé. Début entrainement...")
        model = build_model(input_shape=img_shape, num_classes=NB_CLASS)
        model.summary()

        # Entrainement du modèle
        start = time.time()
        history = train_model(model, X_train, y_train, X_val, y_val, epochs=EPOCHS)
        end = time.time()
        print(f"Temps d'éxécution: {end-start:.2f}s\n")
        
        evaluate_model(model, X_test, y_test)
        graph_plot_save(history)
    
    # Affichage des prédictions 
    print("\nPrédiction sur X_test...")
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
from model_building import *
from y_label_generator import *

import matplotlib.pyplot as plt




if __name__ == "__main__":
    print(f"\nTensorflow version : {tf.__version__}") #2.18.0
    print(f"Keras version : {tf.keras.__version__}") #3.8.0
    
    
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
    X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(images, y, filenames, test_size=0.20)
    print("Nombre d'images dans le Training set: ",len(X_train))
    print("Nombre d'images dans le Test set: ",len(X_test))
    img_shape = X_train[0].shape
    
    # On n'entraine le modele que si l'utilisateur le souhaite
    MODEL_PATH = "best_model.keras"
    EPOCHS = 25
    NB_CLASS = encoded_labels.shape[1]
    
    print("\nDébut entrainement...")
    # Génération du réseau
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
        

    # Create a DataFrame with filenames and predicted labels
    df_results = pd.DataFrame({
        "Filename": filenames_test,
        "Predicted Label": all_predicted_labels
    })

    # Save to CSV   
    df_results.to_csv("predictions.csv", index=False)

    print("Predictions saved to predictions.csv!")


    # imgplot = plt.imshow(X_test[i])
    # plt.show()


    # #subplot(r,c) provide the no. of rows and columns
    # f, axarr = plt.subplots(4,3)

    # for i in range(len(X_test)):
    #     axarr[i].plot(X_test[i])
    #     axarr[i].title.set_title(f"X_test[{i}] prédit {all_predicted_labels[i]}")
    # print(f"Distribution de probabilité pour X_test[0] : {y_test_pred[0]}")

    
    #plotting Loss
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Précision modèle')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.savefig('Loss and validation loss plot.png')
    print("GRAPH SAVED !")
    
    # #plotting PSNR
    # plt.plot(history.history['psnr'])
    # plt.plot(history.history['val_psnr'])
    # plt.title('Model PSNR')
    # plt.ylabel('PSNR')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
import os
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import OneHotEncoder

def get_sample_name(filename):
    """Extracts the sample name from an image filename."""
    match = re.match(r"(SLG?\d+[A-Za-z]*\d*)_BW_([A-Za-z]+).tif", filename)
    if match:
        return match.groups()  # (sample_name, mineral)
    return None

def classify_rock_type(position):
    # On triche un peu sur la position du coeur en l'élargissant légèrement
    """Assigns rock type based on position.

    Args:
        position (float): Position of the taken sample

    Returns:
        str: Roche correspondant à la position
    """
    if (0 <= position <= 33) or (67 <= position <= 72):
        return "Gneiss"
    elif (33 <= position < 44.9) or (49 <= position < 67):
        return "Roche transitoire"
    elif (44.9 <= position < 49):
        return "Core"
    return None

def generate_labels(image_folder, csv_path):
    """Generates Y labels for all images in the dataset.

    Args:
        image_folder (str): _description_
        csv_path (str): _description_

    Returns:
        (pd.Dataframe, array, OneHotEncode): _description_
    """
    df = pd.read_csv(csv_path)
    df["Sample"] = df["Sample"].astype(str)  # Ensure sample names are strings
    df["Position"] = df["Position"].astype(float)  # Ensure positions are doubles
    sample_to_position = dict(zip(df["Sample"], df["Position"]))
    
    data = []
    
    for root, _, files in os.walk(image_folder):
        for filename in files:
            if filename.endswith("Fe.tif") \
            or filename.endswith("Ca.tif") \
            or filename.endswith("Si.tif") \
            or filename.endswith("Pb.tif") \
            or filename.endswith("Ti.tif") :
                result = get_sample_name(filename)
                if not result:
                    continue
                sample_name, mineral = result
                
                # Infer subsample mapping
                matched_sample = None
                for key in sample_to_position:
                    if sample_name.startswith(key) and not re.match(f"{key}\\d", sample_name):
                        matched_sample = key
                        break
                
                if matched_sample:
                    position = sample_to_position[matched_sample]
                    rock_type = classify_rock_type(position)
                    if rock_type:
                        data.append((sample_name, mineral, rock_type))
    
    df_labels = pd.DataFrame(data, columns=["Sample Name", "Mineral", "Rock Type"])
    
    # One-hot encoding
    encoder = OneHotEncoder()
    df_labels["combined_label"] = df_labels["Mineral"].astype(str) + "_" + df_labels["Rock Type"].astype(str)

    # encoded_labels = encoder.fit_transform(df_labels[["combined_label"]]).toarray()
    encoded_labels = encoder.fit_transform(df_labels[["Mineral", "Rock Type"]]).toarray()

    return df_labels, encoded_labels, encoder

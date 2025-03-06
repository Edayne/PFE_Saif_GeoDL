# Purpose
This is an attempt at creating a Deep Learning model to **classify rock sample images** taken around a fracture. 
The idea is that when a fracture is formed in a rock formation, smaller fractures will spread around that one. All of these fractures allow the passage of brine, and is the main source of geothermic power. The passage of brine allows for changes in mineral composition within the fracture, thus X-Ray analyses of minerals will show a clear difference between the outside rock (the **gneis**) and the **core** of the fracture.

# Prerequisites 
Activate a virtual environment with whichever tool you prefer, I used Conda.

```sh
conda init
conda create --name geodl python=3.12
conda activate geodl
pip install -r requirements.txt
```

ATTENTION :
If you do not have a GPU, you will have to do the following after the previous commands.

```sh
pip uninstall tensorflow
pip install tensorflow-cpu
```

# Usage

Make sure you have the necessary images to train the model before executing the program.

```sh
clone https://github.com/Edayne/PFE_Saif_GeoDL
cd PFE_Saif_GeoDL
python main.py
```

# Thanks

This project is made with the blessings of Phd. student Benjamin Avakian and Dr. Karima El Ganaoui. Thanks to the both of them for their help.
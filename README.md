# FoodX-251

```
Sandro Erba
Elia Gaviraghi
```

## Folder Structure 

```
┣ 📂 progetto \                                # Root directory
┣ ┣ 📂 dataset\                                # training and test set CSV
┣ ┃ ┣ 📄 train_info_clean.csv                  # training set CSV after cleaning 
┣ ┃ ┣ 📄 train_info_dirty.csv                  # original CSV
┣ ┃ ┣ 📄 val_info.csv                          # test set CSV
┣ ┣ 📂 matlab_script\                 
┣ ┃ ┣ 📄 alex_net_on_validation.m
┣ ┃ ┣ 📄 clean_and_augment_dataset.m
┣ ┃ ┣ 📄 clustering_features.m                 # script for cleaning dirty dataset
┣ ┃ ┣ 📄 from_folder_to_subfolder.m            # obtain foldering dataset from the original dataset
┣ ┃ ┣ 📄 from_subfolder_to_one_folder.m        # the inverse process
┣ ┣ 📂 notebook\                               # colab files
┣ ┃ ┣ 📄 MobileNetV3.ipynb                     # main notebook where we fine-tuning the selected model
┣ ┃ ┣ 📄 Category_Search_foodx251.ipynb        # category search with selected method using our fine-tuned model
┣ ┃ ┣ 📄 Category_Search_foodx251_cheat.ipynb  # a little bit alternative method for category search
┣ ┃ ┣ 📄 Category_Search_imagenet.ipynb        # inital and alternative method with CNN trained on ImageNet
┣ ┣ 📄 Presentazione Visual.pdf
┣ ┣ 📄 Presentazione Visual.pptx
┣ ┣ 📄 README.md                               # it's me!
```

## 🛠 Instructions for use
zip files reside in a folder owned by the authors. If you want access, please contact the authors of the project.
If you have the .zip files, locate them inside the 'dataset' folder and **check the correspondence in the folder names in the code.**

### ✅ Evaluate our model on test set
if you want to put our network to the test set to the test, you can open the MobileNetV3.ipynb notebook with Colab (or similar) and run the sections:
1. Preliminary options
2. Dataset
3. Data augmentation

Skip the fine-tuning section of the MobileNetV3 model and go directly to the Test set section and run it.

### 🔎 Use category search
Open the notebook Category_Search_foodx251.ipynb and make sure to execute all cell:
1. Preliminary operations
2. Dataset
3. Upload image
4. Category search

Please **don't run the Feature extraction cell** the features on the entire training set have already been calculated beforehand and are loaded automatically in the Preliminary Operations cell, the code is only for demonstration purposes.

Now open Upload Image cell, and you can upload a picture of food in .jpg format and enter the category (label) to which the 251 classes belong.
If you do not know the label to which your image belongs and you do not want an objective evaluation of the results, but only of similar images, enter a random number and disregard the percentages in the output.

#### Automated test category search
If you want to conveniently test our category search on an entire folder, you can use this code cell.
We currently use the project's test set, but by appropriately changing the directory and uploading your folder to the Colab session, you can customise this code ad hoc.

Please note, however, that the uploaded images must have a label and that these labels must be retrievable from a CSV with a similar structure to the one we use, otherwise you would have to modify the code.

## ⛔ Matlab file
these files were used for a 'pre-stage' of the project realisation and the results are already available on the repository, so avoid running them unless you know exactly what you are doing.


## ⚙️ Software and Hardware Requirement
We recommended:
 - Google Colab (or Kaggle, but be sure to adapt the code) with T4 GPU or higher to train the model 
 - alternatively, a PC with Nvidia T4 GPU or higher to train the model, and Python 3
 - MATLAB R2023b with Deep Learning Toolbox, Image Processing Toolbox and Statistics and Machine Learning Toolbox.

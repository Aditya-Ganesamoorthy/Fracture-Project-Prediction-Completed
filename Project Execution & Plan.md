**PROJECT EXECUTION:**



1\) First Download the, 15.14 GB Dataset Zip file from the source

2\) Unzip it

3\) Then, in the project folder merge all the 4 image parts into 1 folder (20,327) images

4\) The same way for labels

5\) Store it in folder as below:



'Bone Fracture Project\\GRAZPEDWRI-DX\_dataset\\data\\images' all merged images

'Bone Fracture Project\\GRAZPEDWRI-DX\_dataset\\data\\labels' all merger labels

'Bone Fracture Project\\GRAZPEDWRI-DX\_dataset\\data\\dataset' store the dataset.csv here



6\) Create split.py and meta.yaml, cfg.yaml file and store there.



'Bone Fracture Project\\GRAZPEDWRI-DX\_dataset\\data\\meta.yaml'

'Bone Fracture Project\\GRAZPEDWRI-DX\_dataset\\data\\cfg.yaml'

'Bone Fracture Project\\split.py'



7\) Now, run split.py. It splits and moves all the merged images and labels into 3 folders train, val, test.

That is, 20,327 images and labels as (14,170 train), (2067 test) and (4090 as valid).



8\) Now, new folders created and looks like:



'Bone Fracture Project\\GRAZPEDWRI-DX\_dataset\\data\\images\\' has '\\test', '\\val', and '\\test\\'

'Bone Fracture Project\\GRAZPEDWRI-DX\_dataset\\data\\labels\\' has '\\test', '\\val', and '\\test\\'



All the merged images and labels are moved into above folders by split.py



9\) Also 3 new csv datasets are created in the 'Bone Fracture Project\\GRAZPEDWRI-DX\_dataset\\' folder.



'Bone Fracture Project\\GRAZPEDWRI-DX\_dataset\\train\_data.csv'

'Bone Fracture Project\\GRAZPEDWRI-DX\_dataset\\test\_data.csv'

'Bone Fracture Project\\GRAZPEDWRI-DX\_dataset\\val\_data.csv'



10\) Now, create augmentation.py file in 'Bone Fracture Project\\augmentation.py'. It will creates augmented images and labels on training set.

That is, train set has 14,170 data. It creates (14,170 \* 2 = 28,340) images and labels and stores it in 'train\_aug' without modifying existing train images and labels folders.



11\) Command to run augmentation.py:



'python augmentation.py --input\_img ./GRAZPEDWRI-DX\_dataset/data/images/train/ --output\_img ./GRAZPEDWRI-DX\_dataset/data/images/train\_aug/ --input\_label ./GRAZPEDWRI-DX\_dataset/data/labels/train/ --output\_label ./GRAZPEDWRI-DX\_dataset/data/labels/train\_aug/'



12\) It creates 2 new folders as below:



'Bone Fracture Project\\GRAZPEDWRI-DX\_dataset\\data\\images\\train\_aug'

'Bone Fracture Project\\GRAZPEDWRI-DX\_dataset\\data\\labels\\train\_aug'



13\) Now, download the yolov7 model and train the model uing this command:



'python export.py --weights yolov7-p6-bonefracture.pt --img 640 --batch 1 --device cpu --simplify --include onnx'





\###################################################################################################



**PROJECT STRUCTURE:**





Bone Fracture Project/

│

├── GRAZPEDWRI-DX\_dataset/

│   │

│   ├── dataset.csv                  ← (converted from your Excel)

│   ├── train\_data.csv               ← (auto-created by split.py)

│   ├── valid\_data.csv               ← (auto-created by split.py)

│   ├── test\_data.csv                ← (auto-created by split.py)

│   │

│   └── data/

│       │

│       ├── cfg.yaml

│       ├── meta.yaml

│       │

│       ├── images/

│       │   ├── train/               ← (created by split.py)

│       │   ├── valid/               ← (created by split.py)

│       │   ├── test/                ← (created by split.py)

│       │   └── train\_aug/           ← (created by augmentation.py)

│       │

│       └── labels/

│           ├── train/               ← (created by split.py)

│           ├── valid/               ← (created by split.py)

│           ├── test/                ← (created by split.py)

│           └── train\_aug/           ← (created by augmentation.py)

│

├── split.py

├── augmentation.py

├── requirements.txt

├── README.md

│

├── runs                             ← (auto-created after model training)

│   ├── train/

│   |    └── yolov7/

│   └── test

│       ├── eval\_test/

│       ├── eval\_train/

│       └── eval\_val/

│

├── web\_app.py

│

└── yolov7-p6-bonefracture.onnx    ← (auto-created after model training)







\#########################################






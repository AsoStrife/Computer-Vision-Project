# 1. Computer-Vision-Project

## 1.1 Goal of the project

The goal of this project was develop a Face Recognition app using an **Local Binary Pattern** approach. After that we used the same approach in order to perform a real time Face Recognition app.   

This project was developed for [Computer Vision](http://people.unica.it/giovannipuglisi/didattica/insegnamenti/?mu=Guide/PaginaADErogata.do;jsessionid=CBB39621933B1A5C549359BBEFDCA119.jvm1?ad_er_id=2017*N0*N0*S2*26520*20168&ANNO_ACCADEMICO=2017&mostra_percorsi=S&step=1&jsid=CBB39621933B1A5C549359BBEFDCA119.jvm1&nsc=ffffffff0909189545525d5f4f58455e445a4a42378b) course at [University of Cagliari](http://corsi.unica.it/informatica), supervised by prof. [Giovanni Puglisi](http://people.unica.it/giovannipuglisi/).

## 1.2 Used tools

Most of the project has been developed using Python as programming language, open source libraries and datasets. In particular it's been used:

- Python 2.7.14
- OpenCV
- Sklearn
- YaleFaces dataset
- Pil

Non open source tools: 

- Matlab

## 1.2 Face Recognition problem

Over the last ten years or so, face recognition has become a popular area of research in computer vision and one of the most successful applications of image analysis and understanding. A facial recognition system is a technology capable of identifying or verifying a person from a digital image or a video frame from a video source. 
For a human is very easy to perform the face recognition process but the same process is not easy for a computer. Computers hahave to deal with numerous interfering factors related to
treatment of images such as: wide chromatic variations, different and different angles of view.
Beyond this there are other factors that can be affect the face recognition process like: occlusion, hair style, glasses, facial expression etc. 

There are multiples methods in which facial recognition systems work, but in general, they work by comparing selected facial features from given image with faces within a database.

## 1.4 Local Binary Pattern

**Local binary patterns (LBP)** is a type of visual descriptor used for classification in computer vision. In this project **LBP** operator was used to filter the features of facial textures. 

LBP transforms image blocks into an array of labels. Such labels (or their statistics, for example histograms) are used as features. Various versions have been developed in the state of the art. 

In the basic version of the LBP we consider the values of a 3x3 pixel neighborhood. For each pixel in a cell, compare the pixel to each of its 8 neighbors (on its left-top, left-middle, left-bottom, right-top, etc.). Follow the pixels along a circle, i.e. clockwise or counter-clockwise. Where the center pixel's value is greater than the neighbor's value, write "0". Otherwise, write "1". This gives an 8-digit binary number (which is usually converted to decimal for convenience).

In its generalized version, each pixel is considered in a circular neighborhood of **P** points with radius **r**. If the elements do not fall on a single pixel, their values are obtained by interpolation. Generally the number of neighboring pixels chosen is 8.	

## 1.5 Fisherface

...

# 2. Project structure

In this chapter we'll be discuss the project structure and his general function.

```
Computer-Vision-Project
│   LICENSE
│   README.md 
│   main.py 
│   rotate.py 
│
└─── algorithms
│    	LBP.py
|
└─── _datasets
│   	YaleFaces.zip
|		YaleFaces_small.zip
|
└─── model
│   	(here will be knn/naivebayes/svm.pkl)
|
└─── utils
│   	dataset.py
│   	utils.py
└─── Matlab (subproject)
│   	gauss.m
│   	gaussianfilter.m
│   	main.m
│   	mask.mat
│   	preproc2.m
└─── Matlab (subproject)
|		utils
│   	│   	utils.py
|		------------------
│   	create_data.py
│   	face_recognize.py
│   	haarcascade_frontalface_default.xml
```

The folder starting with a capital letter rappresent a sub-project, the other folder are used to store the different libraries. 

In the **root** there are the `algorithms` folder, where there is the basic Local Binary Pattern implementation made by myself.  The `dataset` folder contain the zip of the datasets used to test the code. In the `model` folder will be saved training data values (the model), in order to perform the prediction without training the classifier again.  In the `utils` folder there are some helper function.

The **Matlab** folder contain some scripts that perform a illumination normalization. I've not wrote this script, I've just modified in order to work with YaleFaces Dataset.
For more information about the scripts and the authors check their paper: [Enhanced Local Texture Feature Sets for Face Recognition under Difficult Lighting Conditions](http://parnec.nuaa.edu.cn/xtan/paper/TIP-05069-2009.R1-double.pdf).

Authors page: [Xiaoyang Tan's Publications](http://parnec.nuaa.edu.cn/xtan/Publication.htm) 

The **Realtime** folder contain the sub project that perform a real time face recognition using the pc camera. It contain two main scripts: `create_data.py` that generate the dataset shooting some photos using the camera. `face_recognize.py` trainis the classifier and perform the realtime face recognition. More detail will be given in the next chapters. 

# 3. Run the project 

## 3.1 Install dependencies

In order to run, test and modify the source code you must install the following packages.

```shell
#CV2
sudo apt-get install python-opencv

#PIP
sudo apt-get install python-pip
pip install Pillow

# SKLEARN
pip install -U scikit-learn

sudo apt-get install build-essential python3-dev python3-setuptools 
sudo apt-get install python3-numpy python3-scipy
sudo apt-get install libopenblas-dev

# DLIB
sudo apt-get install build-essential cmake
sudo apt-get install libgtk-3-dev
sudo apt-get install libboost-all-dev

# OTHER
sudo pip install scikit-image
sudo pip install dlib
```

## 3.2 Run Main project

First of all you have to unzip the YaleFaces.zip and YaleFaces_small. 

Launch the `python main.py` with the following parameters: 

```shell
--dataset [datasetName]
--classifier [svm, knn, naivebayes]
--training
--histEq
--output

```

By default launching only `python main.py` the script use  YaleFaces dataset without training (loading from model folder). If there are not model in your model folder you must use `--training`. 
`--histEq` is helpful because perform an histogram equalization before calculating the LBP.
If `--output` is setted the script will produce inside `./datasets/your_chosen_algorithm_/your_chosen_dataset/` the PNG of the LBP calculated for each image.

With `--classifier` you can choose which classifier to use between SVM, KNN and NaiveBayes (obviously for each classifier you must perform the training of the model).

By default LBP is performed splitting the images in 12x12 blocks.

#### Usage example

`python main.py --dataset YaleFaces --classifier svm`

`python main.py --training --output --histEq` 



## 3.3 Run Real Time project

.. 



## 3.4 Run Matlab project

...

# 4. Benchmark and classification

Below you will find the results I obtained using my PC (Dell XPS 15 9550, with Intel i7 6700HQ Skylake @ 2.60GHz)

For the classification task I've used `LinearSVM`

| Dataset         | LBP time     | Training time | Testing time | Average Accuracy |
| --------------- | ------------ | ------------- | ------------ | ---------------- |
| YaleFaces       | ≈ 31 seconds | ≈ 33.1        | ≈ 37         | ≈ 0.98           |
| YaleFaces_small | ≈ 4 seconds  | ≈ 4 seconds   | ≈ 4 seconds  | ≈ 0.99           |

| Dataset         | ELBP time | Training time | Testing time | Average Accuracy |
| --------------- | --------- | ------------- | ------------ | ---------------- |
| YaleFaces       |           |               |              |                  |
| YaleFaces_small |           |               |              |                  |

![](http://i67.tinypic.com/msykqq.png)
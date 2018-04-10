#ELBP-Computer-Vision

The goal of this project was develop an ELBP implementation in order to perform a Face Recognition.  

This project was developed for [Computer Vision](http://people.unica.it/giovannipuglisi/didattica/insegnamenti/?mu=Guide/PaginaADErogata.do;jsessionid=CBB39621933B1A5C549359BBEFDCA119.jvm1?ad_er_id=2017*N0*N0*S2*26520*20168&ANNO_ACCADEMICO=2017&mostra_percorsi=S&step=1&jsid=CBB39621933B1A5C549359BBEFDCA119.jvm1&nsc=ffffffff0909189545525d5f4f58455e445a4a42378b) course at [University of Cagliari](http://corsi.unica.it/informatica), supervised by prof. [Giovanni Puglisi](http://people.unica.it/giovannipuglisi/).

For more information about ELBP read the original paper at this link: [Extended local binary patterns for texture classification](https://www.sciencedirect.com/science/article/pii/S0262885612000066).



# Project structure

This project has the following structure

```
ELBP-Computer-Vision
│   README.md
│   main.py    
│
└─── algorithms
│    	ELBP.py
│    	LBP.py
|
└─── _datasets
│   	YaleFaces.zip
|
└─── model
|
└─── utils
│   	dataset.py
│   	utils.py
```

In the `algorithms` folder you can find the basic LBP implementation and the ExtendedLBP implementation. 

In the `dataset` folder there are the zip of the dataset used to test the code. You can add here your dataset.

In the `model` folder, the script store the training data values, in order to perform the prediction without traing the classifier again. 

In the `utils` folder there are some helper function.

#Execute 

First of all you have to unzip the YaleFaces.zip in the same folder. 

You can launch the `python main.py` with the following parameters: 

```shell
python main.py --dataset datasetName
python main.py --algorithm LBP or ELBP
python main.py --training
```

By default launching only `python main.py` the script use LBP algorithm with YaleFaces dataset without training (loading from model folder). If there are not model in your model folder you must user `--training`. 

#### Example

`python main.py --dataset YaleFaces --algorithm LBP`

`python main.py --algorithm ELBP --training`

# Dependencies

In order tu run, test and modify the source code you must install the following package.

```shell
sudo apt-get install python-pip
pip install Pillow

# SKLEARN
sudo apt-get install build-essential python3-dev python3-setuptools 
sudo apt-get install python3-numpy python3-scipy
sudo apt-get install libopenblas-dev

# DLIB
sudo apt-get install build-essential cmake
sudo apt-get install libgtk-3-dev
sudo apt-get install libboost-all-dev
wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
sudo pip install scikit-image
sudo pip install dlib
```


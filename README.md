# Resnet_v2_50 on cat-vs-dog

The code uses predefined models from tensorflow's offical model zoo [tensorflow slim](https://github.com/tensorflow/models/tree/master/research/slim).

To run it, you need to do the following steps

- download the dataset from [kaggle](https://www.kaggle.com/c/dogs-vs-cats/data), and extract it to data folder as below
- run create_label.py to generate label.csv, which contains the filename and it's label
- run train.py to train the model

```bash
python train.py --help
usage: A script to train resnet_2_50 [-h] [--batchsize BATCHSIZE] [--lr LR]
                                     [--numepochs NUMEPOCHS]
                                     [--testsize TESTSIZE]
                                     [--labelmap LABELMAP]
                                     [--numthreads NUMTHREADS]
                                     [--logdir LOGDIR]

optional arguments:
  -h, --help            show this help message and exit
  --batchsize BATCHSIZE
                        batch size
  --lr LR               learning rate
  --numepochs NUMEPOCHS
                        number of epochs to train
  --testsize TESTSIZE   ratio of validation data
  --labelmap LABELMAP   labelmap file
  --numthreads NUMTHREADS
                        number of threads to read data
  --logdir LOGDIR       log directory

```

## structure of data directory

The directory of the dataset is discribed as below

- data
  - test1
    - 1.jpg
    - 2.jpg
    - ...
    - XXXX.jpg
  - train
    - cat.1.jpg
    - cat.2.jpg
    - ...
    - cat.XXXX.jpg
    - dog.1.jpg
    - dog.2.jpg
    - ...
    - dog.XXXX.jpg
  - sampleSubmission.csv

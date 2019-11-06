## Get start
Firstly, you need to run 
``` bash data/download_datasets.sh horse2zebra ```
to get the `horse2zebra`dataset.
the dataset will be downloaded at `data/horse2zebra`with structure

```
horse2zebra
    ├── testA
    ├── testB
    ├── trainA
    └── trainB
```
To run the model

```
python3 train.py
```
The parameters we need are all set in the ```params/params.py```,such like `batch size`,`learing rate`,`dataset path`.

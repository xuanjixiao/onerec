
This is the code implementation of LT4REC.
We demonstrate the algorithm on the dataset gathered from traffic
logs of real world video recommendation system.
## Enviroments

tensorflow==1.12

tensorboard==1.12

pyyaml

absl-py

numpy

sklearn

## Installation

```bash
pip3 install -r requirements.txt
```

## Dataset

For the dataset gathered from traffic logs of real world video recommendation system 
is not completely public, we just provide a small part of the dataset.The data is encrypted and stored as the tfrecord format, with feature columns defined as:
```
features{
    'FEA_SVID': tf.io.FixedLenFeature([1], tf.string,"-1"),
    'FEA_SCP': tf.io.FixedLenFeature([1], tf.string,"-1"),           
    'FEA_SICAT1': tf.io.FixedLenFeature([1], tf.string,"-1"),          
    'FEA_SICAT2': tf.io.FixedLenFeature([1], tf.string,"-1"),          
    'FEA_SIKEYWORD': tf.io.FixedLenSequenceFeature(shape=[], dtype=tf.string,allow_missing=True,default_value="-1"),  
    'FEA_STAG': tf.io.FixedLenSequenceFeature(shape=[], dtype=tf.string,allow_missing=True,default_value="-1"),
    'FEA_VID': tf.io.FixedLenFeature([1], tf.string,"-1"),   
    'FEA_CP': tf.io.FixedLenFeature([1], tf.string,"-1"),   
    'FEA_ICAT1': tf.io.FixedLenFeature([1], tf.string,"-1"),   
    'FEA_ICAT2': tf.io.FixedLenFeature([1], tf.string,"-1"),  
    'FEA_IXFTAG': tf.io.FixedLenSequenceFeature(shape=[], dtype=tf.string,allow_missing=True,default_value="-1"),  
    'FEA_AGE': tf.io.FixedLenFeature([1], tf.string,"-1"),
    'FEA_IKEYWORD': tf.io.FixedLenSequenceFeature(shape=[], dtype=tf.string,allow_missing=True,default_value="-1"),  
    'FEA_DURATION': tf.io.FixedLenFeature([1], tf.string,"-1"),
    'FEA_ALGID': tf.io.FixedLenFeature([1], tf.string,"-1"),
    'FEA_ORDER': tf.io.FixedLenFeature([1], tf.string,"-1"),   
    'FEA_COVERQUALITY': tf.io.FixedLenFeature([1], tf.string,"-1"),
    'FEA_UID': tf.io.FixedLenFeature([1], tf.string,"-1"),
    'FEA_UGROUP': tf.io.FixedLenFeature([1], tf.string,"-1"), 
    'FEA_UCAT': tf.io.FixedLenSequenceFeature(shape=[], dtype=tf.string,allow_missing=True,default_value="-1"),
    'seqnum': tf.io.FixedLenFeature([1], tf.string,"-1"),
    'pctr': tf.io.FixedLenFeature([1], tf.string,"-1"),
    'cvrlabel': tf.io.FixedLenFeature([1], tf.float32,0),
    'label': tf.io.FixedLenFeature([1], tf.int64,0),
    }
```
## Training
Enter the piplines folder.
##### get the mask of single task:
ctr task:

```python main.py --training_mode=single --task_id=0```

cvr task:

```python main.py --training_mode=single --task_id=1```

Your can adjust the nubmer of pruning iteriterations and final remaining parameter rate by seting the following parameters:

```--prune_pruning_iter & --prune_final_rate```

Then you can get the mask of every task. According to the metrics, choose the best mask of every task and rename them to 0.pkl and 1.pkl.

After that, You should create a mask folder named mask under the save_mtl_checkpoints_dir folder.

Finally, move the masks to the mask folder. 

##### traing the multi-task:
```python main.py --training_mode=mtl```
The complete checkpoints file path is as follows：
```
.
├── checkpoints
│   ├── mtl
│   │   ├── checkpoint
│   │   ├── ckpt_epoch-1.data-00000-of-00001
│   │   ├── ckpt_epoch-1.index
│   │   ├── ckpt_epoch-1.meta
│   │   └── mask
│   │       ├── 0.pkl
│   │       └── 1.pkl
│   └── single
│       └── chk_points
│           ├── 0
│           │   ├── checkpoint
│           │   ├── ckpt_init-0.data-00000-of-00001
│           │   ├── ckpt_init-0.index
│           │   ├── ckpt_init-0.meta
│           │   └── mask
│           │       ├── 1_66.87%.pkl
│           │       ├── 2_44.72%.pkl
│           │       ├── 3_29.91%.pkl
│           │       ├── 4_20.0%.pkl
│           │       └── 5_20.0%.pkl
│           └── 1
│               ├── checkpoint
│               ├── ckpt_init-0.data-00000-of-00001
│               ├── ckpt_init-0.index
│               ├── ckpt_init-0.meta
│               └── mask
│                   ├── 1_66.87%.pkl
│                   ├── 2_44.72%.pkl
│                   ├── 3_29.91%.pkl
│                   ├── 4_20.0%.pkl
│                   └── 5_20.0%.pkl
```




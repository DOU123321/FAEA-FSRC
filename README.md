# FAEA-FSRC
The source code of paper 《Function-words Adaptively Enhanced Attention Networks for Few-Shot Inverse Relation Classification》, accepted to IJCAI 2022.


## Requirements
- ``python 3.6``
- ``PyTorch 1.7.0``
- ``transformers 4.0.0``
- ``numpy 1.19``

## Datasets
We experiment our model on two few-shot relation extraction datasets,
 1. [FewRel 1.0](https://thunlp.github.io/1/fewrel1.html)
 2. [FewRel 2.0](https://thunlp.github.io/2/fewrel2_da.html)
 
Please download data from the official links and put it under the ``./data/``. 



## Evaluation
Please download trained model from [here](https://pan.baidu.com/s/14_zbQfVevYi6NxjcIdBEkw) [usoa] and put it under the ``./checkpoint/``. To evaluate our model, use command

**FewRel 1.0**

Under 10-way-1-shot setting
```bash
python train.py \
    --N 10 --K 1 --Q 1 --test_iter 10000\
    --only_test True --load_ckpt "./checkpoint/FAEA-bert-train_wiki-val_wiki-10-1.pth.tar"
```
Under 5-way-1-shot setting
```bash
python train.py \
    --N 5 --K 1 --Q 1 --test_iter 10000\
    --only_test True --load_ckpt "./checkpoint/FAEA-bert-train_wiki-val_wiki-5-1.pth.tar"
```

## Training
**FewRel 1.0**

To run our model, use command

```bash
python train.py
```

This will start the training and evaluating process of FAEA in a 10-way-1-shot setting. You can also use different args to start different process. Some of them are here:

* `train / val / test`: Specify the training / validation / test set.
* `trainN`: N in N-way K-shot. `trainN` is the specific N in training process.
* `N`: N in N-way K-shot.
* `K`: K in N-way K-shot.
* `Q`: Sample Q query instances for each relation.

There are also many args for training (like `batch_size` and `lr`) and you can find more details in our codes.

**FewRel 2.0**

Use command
```bash
python train.py \
    --val val_pubmed --test val_pubmed --ispubmed True 
```


## Results

**FewRel 1.0**

|                   | 5-way-1-shot | 5-way-5-shot | 10-way-1-shot | 10-way-5-shot |
|  ---------------  | -----------  | ------------- | ------------ | ------------- |
| Val               | 90.81 | 94.24 | 84.22 | 88.74 |
| Test              | 95.10 | 96.48 | 90.12 | 92.72 |

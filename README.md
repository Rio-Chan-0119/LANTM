# Tutorial
This repository contains the implementation for our paper "**Supervised Neural Topic Modeling with Label Alignment**".

## 0. Environment
```
torch==1.12.0
scipy==1.10.1
sklearn==1.2.1
pyyaml==6.0.2
```

## 1. Training
Train LANTM+ECRTM on 20NG with K=50 by default arguments.
```shell
python main.py --mode "train"
```
On completing this, there will be 4 new files under `output/20NG/LANTM_ECRTM/config1`:
- `label_topic_mat.npy`: the Numpy ndarray file of $\boldsymbol{\lambda}$
- `param.pt`: the trained parameters of LANTM+ECRTM
- `topic_word_dist.npy`: the Numpy ndarray file of $\boldsymbol{\beta}$
- `topic_words.txt`: top-10 topic words of each topic


## 2. Evaluation
The evaluation depends on `output/20NG/LANTM+ECRTM/config1/param.pt`.

```shell
python main.py --mode "eval"
```

By default, our provided code will not calculate $C_V$. However,
we provide the code to calculate it if the user is ready to [use palmetto as java program](https://github.com/dice-group/Palmetto/wiki/How-Palmetto-can-be-used#as-java-program).
To calculate $C_V$, please:
1. Install java.

    `sudo apt install openjdk-11-jdk`
2. Download the [java jar](https://hobbitdata.informatik.uni-leipzig.de/homes/mroeder/palmetto/palmetto-0.1.0-jar-with-dependencies.jar) to `./LANTM/palmetto`. It is developed by [palmetto](https://github.com/dice-group/Palmetto).
3. Download and extract [preprocessed Wikipedia articles](https://hobbitdata.informatik.uni-leipzig.de/homes/mroeder/palmetto/Wikipedia_bd.zip) to `./LANTM/palmetto/wikipedia` as the reference corpus.
4. Perform evaluation using

   ```shell
   python main.py --mode "eval" --calculate_CV
   ```

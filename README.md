# DeepQA2
Train and serve QA Model with TensorFlow
[![Join the chat at https://gitter.im/chatbot-pilots/DeepQA](https://badges.gitter.im/chatbot-pilots/DeepQA.svg)](https://gitter.im/chatbot-pilots/DeepQA?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

# DeepQA
Inspired and inherited from [DeepQA](https://github.com/Conchylicultor/DeepQA).

# Pre-process data

## Download data
Sample Corpus http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html

```
cd data_creator

```

# Train 
## Install
First, copy *dataset-creator/dialogues/dataset.pkl* to *data/dataset.pkl*.
Second, install deps, 
```
pip install -r requirements.txt
```

## Configure
```
cp config.sample.ini config.ini # modify keys
```

* data
Store corpus.

* logs
Store logs.

* save
Store generated models.

## Train
```
python train.py
```

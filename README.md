# DeepQA2
[![Join the chat at https://gitter.im/chatbot-pilots/DeepQA](https://badges.gitter.im/chatbot-pilots/DeepQA.svg)](https://gitter.im/chatbot-pilots/DeepQA?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Train and serve QA Model with TensorFlow

Tested with TensorFlow#0.11.0rc2, Python#3.5.

[Install Nvidia Drivers, CUDNn, Python, TensorFlow on Ubuntu 16.04](https://gist.github.com/Samurais/e20a8283708d37f1d7c9a709e9332429)

# DeepQA
Inspired and inherited from [DeepQA](https://github.com/Conchylicultor/DeepQA).

# Install deps
```
pip install -r requirements.txt
```

# Pre-process data
```
cd deepqa2 
cp config.sample.ini config.ini
python dataset/preprocesser.py
```

Sample Corpus http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html

# Train Model

## Configure
```
cp config.sample.ini config.ini # modify keys
python train.py
```


# Serve Model
```

```
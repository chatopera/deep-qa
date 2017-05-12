# deprecated

If you are interested in further enhancements and investigations, just watch *Next* repo.

[https://github.com/Samurais/Neural\_Conversation\_Models](https://github.com/Samurais/Neural_Conversation_Models)

# Approaching a Chatbot Service
[Part 1: Introduction](http://www.leiphone.com/news/201702/O9PGyImfH1Vq3fxV.html)

[Part 2: Bot Engine](http://www.leiphone.com/news/201702/oY07cF3HVIp7Yo1s.html)

[Part 3: Bot Model](http://www.leiphone.com/news/201702/4OZau7OfcNO0v1u5.html)

> This repository is align with  **Part 3: Bot Model**.

# DeepQA2
[![Join the chat at https://gitter.im/chatbot-pilots/DeepQA](https://badges.gitter.im/chatbot-pilots/DeepQA.svg)](https://gitter.im/chatbot-pilots/DeepQA?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Train and serve QA Model with TensorFlow

Tested with TensorFlow#0.11.0rc2, Python#3.5.

[Install Nvidia Drivers, CUDNn, Python, TensorFlow on Ubuntu 16.04](https://gist.github.com/Samurais/e20a8283708d37f1d7c9a709e9332429)

# DeepQA
Inspired and inherited from [DeepQA](https://github.com/Conchylicultor/DeepQA/issues/44).

# Install deps
```
pip install -r requirements.txt
```

# Pre-process data
Process data, build vocabulary, word embedding, conversations, etc.
```
cp config.sample.ini config.ini
python deepqa2/dataset/preprocesser.py
```

Sample Corpus http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html

# Train Model
Train language model with Seq2seq.
```
cp config.sample.ini config.ini # modify keys
python deepqa2/train.py
```

# Serve Model
Provide RESt API to access language model.
```
cd DeepQA2/save/deeplearning.cobra.vulcan.20170127.175256/deepqa2/serve
cp db.sample.sqlite3 db.sqlite3 
python manage.py runserver 0.0.0.0:8000
```

Access Service with RESt API
```
POST /api/v1/question HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json
Authorization: Basic YWRtaW46cGFzc3dvcmQxMjM=
Cache-Control: no-cache

{"message": "good to know"}

response
{
  "rc": 0,
  "msg": "hello"
}
```
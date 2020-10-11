# Note, this repo is deprecated.

If you are interested in further enhancements and investigations, just watch *Next* repo.

[https://github.com/Samurais/Neural\_Conversation\_Models](https://github.com/Samurais/Neural_Conversation_Models)

# Approaching a Chatbot Service
[![Join the chat at https://gitter.im/chatbot-pilots/DeepQA](https://badges.gitter.im/chatbot-pilots/DeepQA.svg)](https://gitter.im/chatbot-pilots/DeepQA?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![Docker Pulls](https://img.shields.io/docker/pulls/samurais/deepqa2.svg?maxAge=2592000)](https://hub.docker.com/r/samurais/deepqa2/) [![Docker Stars](https://img.shields.io/docker/stars/samurais/deepqa2.svg?maxAge=2592000)](https://hub.docker.com/r/samurais/deepqa2/) [![Docker Layers](https://images.microbadger.com/badges/image/samurais/deepqa2.svg)](https://microbadger.com/#/images/samurais/deepqa2)

[![chatoper banner][co-banner-image]][co-url]

[co-banner-image]: https://user-images.githubusercontent.com/3538629/42383104-da925942-8168-11e8-8195-868d5fcec170.png
[co-url]: https://www.chatopera.com

[Part 1: Introduction](http://www.leiphone.com/news/201702/O9PGyImfH1Vq3fxV.html)

[Part 2: Bot Engine](http://www.leiphone.com/news/201702/oY07cF3HVIp7Yo1s.html)

[Part 3: Bot Model](http://www.leiphone.com/news/201702/4OZau7OfcNO0v1u5.html)

> This repository is align with  **Part 3: Bot Model**.

Train and serve QA Model with TensorFlow

Tested with TensorFlow#0.11.0rc2, Python#3.5.

[Install Nvidia Drivers, CUDNn, Python, TensorFlow on Ubuntu 16.04](https://gist.github.com/Samurais/e20a8283708d37f1d7c9a709e9332429)

# DeepQA
Inspired and inherited from [DeepQA](https://github.com/Conchylicultor/DeepQA/issues/44).

# Install deps
```
pip install -r requirements.txt
```

# Install TensorFlow
```
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0rc2-cp35-cp35m-linux_x86_64.whl
pip install —-upgrade $TF_BINARY_URL
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

# Train with Docker
## Install 
* [docker](https://docs.docker.com/engine/installation/linux/ubuntu/)
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
## Train
```
docker pull samurais/deepqa2:latest
cd DeepQA2
./scripts/train_with_docker.sh
```



## Chatopera 云服务

[https://bot.chatopera.com/](https://bot.chatopera.com/)

[Chatopera 云服务](https://bot.chatopera.com)是一站式实现聊天机器人的云服务，按接口调用次数计费。Chatopera 云服务是 [Chatopera 机器人平台](https://docs.chatopera.com/products/chatbot-platform/index.html)的软件即服务实例。在云计算基础上，Chatopera 云服务属于**聊天机器人即服务**的云服务。

Chatopera 机器人平台包括知识库、多轮对话、意图识别和语音识别等组件，标准化聊天机器人开发，支持企业 OA 智能问答、HR 智能问答、智能客服和网络营销等场景。企业 IT 部门、业务部门借助 Chatopera 云服务快速让聊天机器人上线！

<details>
<summary>展开查看 Chatopera 云服务的产品截图</summary>
<p>

<p align="center">
  <b>自定义词典</b><br>
  <img src="https://static-public.chatopera.com/assets/images/64530072-da92d600-d33e-11e9-8656-01c26caff4f9.png" width="800">
</p>

<p align="center">
  <b>自定义词条</b><br>
  <img src="https://static-public.chatopera.com/assets/images/64530091-e41c3e00-d33e-11e9-9704-c07a2a02b84e.png" width="800">
</p>

<p align="center">
  <b>创建意图</b><br>
  <img src="https://static-public.chatopera.com/assets/images/64530169-12018280-d33f-11e9-93b4-9db881cf4dd5.png" width="800">
</p>

<p align="center">
  <b>添加说法和槽位</b><br>
  <img src="https://static-public.chatopera.com/assets/images/64530187-20e83500-d33f-11e9-87ec-a0241e3dac4d.png" width="800">
</p>

<p align="center">
  <b>训练模型</b><br>
  <img src="https://static-public.chatopera.com/assets/images/64530235-33626e80-d33f-11e9-8d07-fa3ae417fd5d.png" width="800">
</p>

<p align="center">
  <b>测试对话</b><br>
  <img src="https://static-public.chatopera.com/assets/images/64530253-3d846d00-d33f-11e9-81ea-86e6d47020d8.png" width="800">
</p>

<p align="center">
  <b>机器人画像</b><br>
  <img src="https://static-public.chatopera.com/assets/images/64530312-6442a380-d33f-11e9-869c-85fb6a835a97.png" width="800">
</p>

<p align="center">
  <b>系统集成</b><br>
  <img src="https://static-public.chatopera.com/assets/images/64530281-4ecd7980-d33f-11e9-8def-c53251f30138.png" width="800">
</p>

<p align="center">
  <b>聊天历史</b><br>
  <img src="https://static-public.chatopera.com/assets/images/64530295-5856e180-d33f-11e9-94d4-db50481b2d8e.png" width="800">
</p>

</p>
</details>


<p align="center">
  <b>立即使用</b><br>
  <a href="https://bot.chatopera.com" target="_blank">
      <img src="https://static-public.chatopera.com/assets/images/64531083-3199aa80-d341-11e9-86cd-3a3ed860b14b.png" width="800">
  </a>
</p>

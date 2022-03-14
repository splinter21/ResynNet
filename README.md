# ResynNet

## Introduction
Refine a video frame based on nearby frames. WIP

## CLI Usage

### Installation

```
git clone git@github.com:hzwer/ResynNet.git
cd ResynNet
pip3 install -r requirements.txt
```

* Download the pretrained model from [here](https://drive.google.com/file/d/1shfl2zKirPrCUcoQmY5EzOv8xLKJ2cR7/view?usp=sharing).

* Unzip and move the pretrained parameters to train_log/\*

### Run

```
python3 inference_img.py --origin example/origin.png --ref example/ref0.png example/ref1.png
```

## Sponsor
Many thanks to [Grisk](https://grisk.itch.io/rife-app).

感谢支持 Paypal Sponsor: https://www.paypal.com/paypalme/hzwer

<img width="160" alt="image" src="https://cdn.luogu.com.cn/upload/image_hosting/5h3609p1.png"><img width="160" alt="image" src="https://cdn.luogu.com.cn/upload/image_hosting/yi3kcwnw.png">

# image2latex_transformer_tensorflow2.x
## 简介
基于tensorflow2.x开发的公式识别项目，encoder使用[MASTER中实现的GCBlock](https://github.com/jiangxiluning/MASTER-TF)，decoder使用transformer-decoder。项目结构简单，易于学习相关技术。


## 细节
*    encoder抽取特征后嵌入2D位置编码
*    为方便batch训练，将图片进行填充255的pad操作
*    图片pad之后嵌入2D位置编码并进行一维拉伸后会产生有效位置编码被割裂的问题，通过将图片进行270度旋转解决此问题

## 训练
- 下载已经预处理完成的latex100k数据：
https://pan.baidu.com/s/18imVP2SaCosNE8oXYKKYVQ  密码: oj2c
- 解压到./dataset下
- python train.py

## Demo
- 下载模型文件：
https://pan.baidu.com/s/1imbE5TWgHgLRqRDJ-AY48A  密码: n90c
- 解压到当前目录下
- python evaluate.py
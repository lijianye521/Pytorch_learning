# 一个能识别1000种事物的一个简单resnet101网络（菜业）

## 环境要求

pytorch

python3.7(其他版本应该也可以)

## 功能

能够实现1000种事物的识别，基本的实现原理是利用已经训练好的resnet进行图像识别，将图像进行预处理，作为网络的输入然后将网络置为eval模式

最后按照概率，将概率最大的事物作为输出

## 代码

```python
# This is a sample Python script.
import torch
from torchvision import models
from PIL import Image
from torchvision import transforms
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   resnet=models.resnet101(pretrained=True)#加载深度学习网络resnet101
   #预处理 在本例中，我们定义了一个预处理函数，将输入图像缩放到 256×256 个像素，围绕中心图像裁剪为 224×224 个像素，并将其转换为一个张量，对其 RGB 分量（红色、绿色和蓝色）进行归一化处理，使其具有定义的均值和标准差
   preprocess = transforms.Compose([
   transforms.Resize(256),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   transforms.Normalize(
         mean=[0.485, 0.456, 0.406],
         std=[0.229, 0.224, 0.225]
   )])
   img = Image.open("dog.jpg")
   print(img)
   img_t = preprocess(img)
   batch_t = torch.unsqueeze(img_t, 0)
   resnet.eval()#我们需要将网络置于 eval 模式。
   out = resnet(batch_t)
   print(out)
   #打开文本文件 将其放进label数组
   with open('imagenet_classes.txt') as f:
      labels = [line.strip() for line in f.readlines()]
   index = torch.max(out, 1) #index代表着张量中最大的值的索引
   print(index)#由此得知索引在index[1]里  所以索引这个数字应该是index[1].item()
   print(index[1])
   percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100  #torch.nn.functional.softmax()可以做归一化处理
   print(percentage[index[1].item()])
   print(labels[index[1].item()])
   #下面是一个排序处理
   print('a sort ------------------------')
   indices = torch.sort(out, descending=True)#降序排序
   x=indices[1][0]
   print(x)
   x=x.numpy()#x是索引可能值的降序排序
   print(x)#将x转化为数组
   for i in range(5):
      item=labels[x[i]]
      per=percentage[x[i]].item()
      print(item)
      print(per)








```


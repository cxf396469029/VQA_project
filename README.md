# VQA_projects

## 研究背景
视觉问答是一个比较新的领域，结合了传统的计算机视觉任务和自然语言处理任务，一个VQA系统将图像和关于图像任何形式的开放式的自然语言问题作为输入，自然语言答案作为输出。

## 研究意义
对视觉问答进行研究，对很多领域都会有巨大的帮助。例如：
作为视力障碍和盲人的辅助工具，帮助他们从看不到的世界中获取信息
针对一堆乱的图片，通过视觉问答筛选我们想要的图片
优化人机交互体验，构建更好的人机交互系统

## 研究方法及过程
提取图片特征 输入一张图片，转换格式输入到VGG模型提取特征
提取文字特征 输入自然语言问题，用Glove模型将其转化为300维度向量
融合特征     将处理好的文字特征放入LSTM中再与图片特征相融合
得到结果     用softmax分类器得到最后的结果

## 开发工具、框架
anaconda、keras、flask

![horse](https://github.com/cxf396469029/VQA_project/raw/master/%E5%9B%BE%E7%89%872.png)
![ball](https://github.com/cxf396469029/VQA_project/raw/master/%E5%9B%BE%E7%89%873.png)
![socks](https://github.com/cxf396469029/VQA_project/raw/master/%E5%9B%BE%E7%89%874.png)

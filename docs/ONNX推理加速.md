目录
1. [理论](#1-理论)

# 1. 理论
工业里，线上部署尤为重要。 如果**深度学习模型推理（预测）可以节省一些GPU资源**、**甚至是仅使用CPU完成**，岂不美哉？

一个**比较成熟的加速推理方案**：ONNX模型格式 + ONNX Runtime

<br>

下面是**onnx模型标准**：

<img height="400" src="images/onnx.png"/>

ONNX提供了**一套标准的AI模型格式**，可以统一传统的机器学习、前沿的深度学习模型。

模型定义使用了protocal buffer。

<br>

**ONNX Runtime**是微软开源的一套**针对ONNX模型标准的推理加速引擎**， 通过**内置的图优化**（Graph  Optimization）和各种**硬件加速**来实现。

下面是**加速效果对比**：

<img height="400" src="images/onnx-speed-example.png"/>

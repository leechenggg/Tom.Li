#!/usr/bin/env python
# coding: utf-8

# 
# # 作业四：撰写项目README并完成开源
# 
# ## 评分标准
# 1.格式规范（有至少3个小标题，内容完整），一个小标题5分，最高20分
# 
# 2.图文并茂，一张图5分，最高20分
# 
# 3.有可运行的代码，且代码内有详细注释，20分
# 
# 4.代码开源到github，15分
# 
# 5.代码同步到gitee，5分
# 
# ## 作业目的
# 使用MarkDown撰写项目并且学会使用开源工具。
# 
# 
# 
# ## 参考资料：
# - [如何写好一篇高质量的精选项目？](https://aistudio.baidu.com/aistudio/projectdetail/2175889)

# 
# * # 基于paddlex钢材表面缺陷检测
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/bc09053d79f6454ab78e1823063ca2a04be7fb45d1c64eecba1d42d18db2cb93)
# ![](https://ai-studio-static-online.cdn.bcebos.com/b37f7b1e531b4f6fb1aa685fffca96345bda5f6d30704f8f85d5cdd417616da8)
# 
# 
# 
# 

#  ## 一、项目背景介绍
# 钢铁表面质量的好坏，直接影响后续产品的制造质量。但在生产过程中，不可避免地会产生一些缺
# 陷。这些缺陷会严重影响产品质量，给企业造成经济损失。因此钢铁进行缺陷检测极为重要。传统的基于机器视觉的表面缺陷检测方法，往往采用常规图像处理算法或人工设计特征加分类器方式。这种方式成本高不够精确，在有充分的数据和高性能硬件支持的条件下，基于深度学习的表面缺陷检测可以大大提高检测精度以及检测效率。今天的项目是钢铁表面的缺陷检测，以下是[工业场景表面缺陷检测数据集及论文集](https://github.com/Charmve/Surface-Defect-Detection/blob/master/ReadmeChinese.md#1%E9%92%A2%E6%9D%90%E8%A1%A8%E9%9D%A2%E6%95%B0%E6%8D%AE%E9%9B%86neu-cls)可能对你有帮助，建议收藏。

# ## 二、数据介绍
# 
# **数据集介绍**
# 
# 地址：http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/04af07309c4e4dafa5f2e6c60739c6dae1124ddaa3fa4f88aa9c7507894f79be)
# 
# 
# 由东北大学（NEU）发布的表面缺陷数据库，收集了热轧钢带的六种典型表面缺陷，即轧制氧化皮（RS），斑块（Pa），开裂（Cr），点蚀表面（ PS），内含物（In）和划痕（Sc）。该数据库包括1,800个200×200像素的灰度图像：六种不同类型的典型表面缺陷，每一类缺陷包含300个样本。对于缺陷检测任务，数据集提供了注释，指示每个图像中缺陷的类别和位置。对于每个缺陷，黄色框是指示其位置的边框，绿色标签是类别分数。
# 
# 

# In[2]:



# 解压数据集到MyDataset文件夹中
get_ipython().system('unzip data/data102850/NEU-DET.zip -d ./MyDataset/')


# ## 三、模型介绍
# PaddleX目前提供了FasterRCNN和YOLOv3两种检测结构，多种backbone模型，可满足开发者不同场景和性能的需求。本项目中采用YOLOv3-MobileNetV3作为检测模型进行钢材缺陷检测。模型优点是模型小，移动端上预测速度有优势。因为之后要部署到移动端所以我选择了这个模型。
# 

# ## 四、模型训练
# PaddleX提供了丰富的视觉模型，通过查阅[PaddleX模型库](https://paddlex.readthedocs.io/zh_CN/release-1.3/appendix/model_zoo.html)，在目标检测中提供了RCNN和YOLO系列模型。在本项目中采用YOLOv3-MobileNetV3作为检测模型进行钢材缺陷检测。

# In[ ]:


import paddlex as pdx
from paddlex import transforms as T


# In[ ]:


# 定义训练和验证时的transforms
# API说明：https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/paddlex/cv/transforms/operators.py
train_transforms = T.Compose([
    T.MixupImage(mixup_epoch=250), T.RandomDistort(),
    T.RandomExpand(im_padding_value=[123.675, 116.28, 103.53]), T.RandomCrop(),
    T.RandomHorizontalFlip(), T.BatchRandomResize(
        target_sizes=[320, 352, 384, 416, 448, 480, 512, 544, 576, 608],
        interp='RANDOM'), T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transforms = T.Compose([
    T.Resize(
        608, interp='CUBIC'), T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# In[ ]:


# 定义训练和验证所用的数据集
# API说明：https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/paddlex/cv/datasets/voc.py#L29
train_dataset = pdx.datasets.VOCDetection(
    data_dir='MyDataset',
    file_list='MyDataset/train_list.txt',
    label_list='MyDataset/labels.txt',
    transforms=train_transforms,
    shuffle=True)

eval_dataset = pdx.datasets.VOCDetection(
    data_dir='MyDataset',
    file_list='MyDataset/val_list.txt',
    label_list='MyDataset/labels.txt',
    transforms=eval_transforms,
    shuffle=False)


# In[ ]:


# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标，参考https://github.com/PaddlePaddle/PaddleX/tree/release/2.0-rc/tutorials/train#visualdl可视化训练指标
num_classes = len(train_dataset.labels)
model = pdx.models.YOLOv3(num_classes=num_classes, backbone='MobileNetV3_ssld')


# In[ ]:


# API说明：https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/paddlex/cv/models/detector.py#L155
# 各参数介绍与调整说明：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html
model.train(
    num_epochs=300,
    train_dataset=train_dataset,
    train_batch_size=2,
    eval_dataset=eval_dataset,
    learning_rate=0.001 / 8,
    warmup_steps=1000,
    warmup_start_lr=0.0,
    save_interval_epochs=20,
    lr_decay_epochs=[216, 243, 275],
    save_dir='output/yolov3_mobilenet')


# ## 五、模型评估
# 该部分主要是对训练好的模型进行评估，可以是用验证集进行评估，或者是直接预测结果。评估结果和预测结果尽量展示出来，增加吸引力。

# In[ ]:


import glob
import numpy as np
import threading
import time
import random
import os
import base64
import cv2
import json
import paddlex as pdx

image_name = 'MyDataset/JPEGImages/pitted_surface_174.jpg'

model = pdx.load_model('output/yolov3_mobilenet/best_model')

img = cv2.imread(image_name)
result = model.predict(img)

keep_results = []
areas = []
f = open('output/yolov3_mobilenet/result.txt','a')
count = 0
for dt in np.array(result):
    cname, bbox, score = dt['category'], dt['bbox'], dt['score']
    if score < 0.5:
        continue
    keep_results.append(dt)
    count+=1
    f.write(str(dt)+'\n')
    f.write('\n')
    areas.append(bbox[2] * bbox[3])
areas = np.asarray(areas)
sorted_idxs = np.argsort(-areas).tolist()
keep_results = [keep_results[k]
                for k in sorted_idxs] if len(keep_results) > 0 else []
print(keep_results)
print(count)
f.write("the total number is :"+str(int(count)))
f.close()


# In[ ]:


pdx.visualize_detection(image_name, result, threshold=0.5, save_dir='./output/yolov3_mobilenet')


# ## 六、总结与升华
# 本文参考作者https://aistudio.baidu.com/aistudio/personalcenter/thirdview/791590 项目来写的，通过本次项目编写，学习了编写项目的流程以及BML使用方法。本人刚开始入门机器学习，还有很多需要学习的地方。感谢创造营的老师，希望以后自己能够独立完成模型的训练与部署。![](https://ai-studio-static-online.cdn.bcebos.com/8f411a2beffb408c9f56054dffa2b7ee6945a414c01a484a97ada7c0172d916b)
# 

# ## 提交链接
# aistudio链接：我在AI Studio上获得白银等级，点亮2个徽章，来互关呀~ https://aistudio.baidu.com/aistudio/usercenter
# 
# github链接：
# 
# gitee链接：

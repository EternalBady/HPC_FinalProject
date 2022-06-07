# HPC-FinalProject
## Title: 一维传热方程的迭代求解
## Information
autior：苏梓鑫;
Data:   2022/06/06;
## Folder Introduction
### analysis
### bin
### lib
### src
## File Introduction

## Log
### 2022.6.1
简单写完了显式的迭代，测试结果是一个大概在$[0,\pi]$的正弦函数形状。
### 2022.6.3
前面写完的显式迭代，结果并行有问题，因为取j+1的时候会涉及到processor通信，导致越界。所以改写成矩阵的乘法形式，重写了显式迭代。
### 2022.6.5
把main.c 变成demo，两种方法分开写，然后测试了显式的结果是正确的，并且可以正确的并行。等全部写完再一起测试结果。
### 2022.6.6
增加注释以及添加计时模块
### 2022.6.7
完成HDF5重启功能

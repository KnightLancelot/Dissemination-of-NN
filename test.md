
## 神经网络工作原理

机器学习并没有什么魔法，它其实就是一个找出过往数据的内在规律，并对未来新数据进行预测的过程，所有的机器学习，包括深度学习，都是找到从输入到输出的最佳拟合函数 的过程，传统机器学习可能用的是一些从统计学中传承的方法，而作为机器学习的一个新领域的深度学习，则是从生物神经网络原理中得到的灵感，用网状结构逐步调整各神经元权重的方法来拟合函数。

# 前言

**What I cannot create, I do not understand.**

**如果我不能亲手搭建起来一个东西，那么我就不能理解它。 -- 美国物理学家理查德·费曼**

本次学习属于科普性质，不涉及到任何代码，仅包括少量数学公式，请放心食用。

想必大家在之前的工作生活中也看到过一些讲解神经网络的视频、文章，里面也往往举出一些（线性）回归拟合的例子，来讲解神经网络的工作原理。


![线性拟合](https://note.youdao.com/yws/api/personal/file/WEB223de80f0450a96492eb5bb9afbfea0e?method=getImage&version=38745&cstk=Xuv5iEhK#pic_right)

![非线性拟合](https://note.youdao.com/yws/api/personal/file/WEBfa590451c6faec9973c420eabd042bfa?method=getImage&version=38744&cstk=Xuv5iEhK#pic_right)


|正向|侧向|
|---|---|
|<img src="https://note.youdao.com/yws/api/personal/file/WEB716ea6a0a65638a21ff61ab84aace16d?method=getImage&version=38746&cstk=Xuv5iEhK"/>|<img src="https://note.youdao.com/yws/api/personal/file/WEB53ecc69191fbd7e4c0242b53a6e70b41?method=getImage&version=38743&cstk=Xuv5iEhK"/>|

然而，大家一定要记住，这只是为了便于大家理解而举的基础原理的例子。
其实，学术界所研究的深度神经网络，往往更具想象力、创造力，并不是用来解决这些小儿科问题的。

|GAN|
|---|
|<img src="https://pic4.zhimg.com/80/v2-f3360c91f6ab4f3fecc2d135a0813e07_720w.webp"/>|
|<img src="https://pic1.zhimg.com/80/v2-87d8f61bfd6c3f5baef9a354bcb141ac_720w.webp"/>|
|<img src="https://img-blog.csdnimg.cn/2019071716365388.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3o3MDQ2MzA4MzU=,size_16,color_FFFFFF,t_70"/>|


|DETR|
|---|
|<img src="https://note.youdao.com/yws/api/personal/file/WEB441c29d5f9589abf02858bbf2846ea03?method=getImage&version=38816&cstk=Xuv5iEhK"/>|
|<img src="https://note.youdao.com/yws/api/personal/file/WEB43d990ca1266556de98f27657e7f0264?method=getImage&version=38815&cstk=Xuv5iEhK"/>|

|MAE|
|---|
|<img src="https://note.youdao.com/yws/api/personal/file/WEBea069d2f57a481b2fe9134c27e8a56c4?method=getImage&version=38782&cstk=Xuv5iEhK"/>|
|<img src="https://note.youdao.com/yws/api/personal/file/WEB6a867e1507ca673e2a7f7f204745bb88?method=getImage&version=38783&cstk=Xuv5iEhK"/>|


多模态的对比学习、迁移学习
|StyleCLIP|
|---|
|<img src="https://note.youdao.com/yws/api/personal/file/WEB391d51990e54217caacd9b052aed883f?method=getImage&version=38823&cstk=Xuv5iEhK"/>|
|<img src="https://note.youdao.com/yws/api/personal/file/WEBaa3570f23053b638b46fa2a7b9a94818?method=getImage&version=38822&cstk=Xuv5iEhK"/>|
|<img src="https://note.youdao.com/yws/api/personal/file/WEB380c43f6a1ab7f7e946dbcc55f0dfd72?method=getImage&version=38824&cstk=Xuv5iEhK"/>|

|CLIFS|
|---|
|<img src="https://note.youdao.com/yws/api/personal/file/WEB61bb43750277f481df7d9383dfaac5ee?method=getImage&version=38825&cstk=Xuv5iEhK"/>|
|<img src="https://note.youdao.com/yws/api/personal/file/WEBe4fa103ebafe863122094b9e74c2333a?method=getImage&version=38826&cstk=Xuv5iEhK"/>|
|<img src="https://note.youdao.com/yws/api/personal/file/WEB5ab2d822db7c194ead8db1863e4468db?method=getImage&version=38821&cstk=Xuv5iEhK"/>|

CLIP+GLIDE/ 扩散模型/ 图片的多样性好于GAN
|unCLIP/ DALL·E 2|
|---|
|<img src="https://theaigang.com/wp-content/uploads/2022/04/Dallle2-theAIgang-01.jpg"/>|
|<img src="https://img-blog.csdnimg.cn/02c1c0a7b8a749c18440c52c44735ade.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5pWw5o2u5a6e5oiY5rS-,size_20,color_FFFFFF,t_70,g_se,x_16"/>|
|<img src="https://note.youdao.com/yws/api/personal/file/WEB4ee0110439e3880bdd26ad238dc8cfee?method=getImage&version=38847&cstk=Xuv5iEhK"/>|



咱们此次学习的重点，并不是对上面那些研究方向的泛泛而谈，而是结合我在学习神经网络时产生的种种困惑、经验、教训，将注意力聚焦在神经网络如何工作这一点上。

在开始讲解原理之前，给大家介绍一下”Tensorflow Playground“，帮助大家在理论认知之前，能对神经网络有一个大体的感性认知。

http://playground.tensorflow.org/

接下来我再给大家讲解一下神经网络大体的工作原理——前向计算、反向传播。


### 1.定义神经网络结构

本例的目的是要用神经网络完成曲线拟合。

根据万能近似定理的要求，我们定义一个两层的神经网络，输入层不算，一个隐藏层，含3个神经元，一个输出层。下图显示了此次用到的神经网络结构。

<img src="https://note.youdao.com/yws/api/personal/file/WEB8300513275665064696013f7995462ca?method=getImage&version=38892&cstk=Xuv5iEhK" />

单入单出的双层神经网络

为什么用3个神经元呢？这也算是最佳实践的结果。因为输入层只有一个特征值，我们不需要在隐层放很多的神经元，先用3个神经元试验一下。如果不够的话再增加，神经元数量是由超参控制的。

#### 输入层

输入层就是一个标量x值（如果是成批输入，则是一个矢量或者矩阵）。

$$
```math
X = (x)
```
$$

#### 权重矩阵W1/B1

$$
```math

W1=
\begin{pmatrix}
w1_{11} & w1_{12} & w1_{13}
\end{pmatrix}

```
$$

$$
```math

B1=
\begin{pmatrix}
b1_{1} & b1_{2} & b1_{3} 
\end{pmatrix}

```
$$

#### 隐层

我们用3个神经元：

$$
```math

Z1 = \begin{pmatrix}
    z1_1 & z1_2 & z1_3
\end{pmatrix}

```
$$

$$
```math

A1 = \begin{pmatrix}
    a1_1 & a1_2 & a1_3
\end{pmatrix}

```
$$


#### 权重矩阵W2/B2

W2的尺寸是3x1，B2的尺寸是1x1。

$$
```math

W2=
\begin{pmatrix}
w2_{11} \\\\
w2_{21} \\\\
w2_{31}
\end{pmatrix}

```
$$

$$
```math

B2=
\begin{pmatrix}
b2_{1}
\end{pmatrix}

```
$$

#### 输出层

由于我们只想完成一个拟合任务，所以输出层只有一个神经元，尺寸为1x1：

$$
```math

Z2 = 
\begin{pmatrix}
    z2_{1}
\end{pmatrix}

```
$$

### 2.前向计算

<img src="https://note.youdao.com/yws/api/personal/file/WEB8663532ba2cb818c7543a75f6bca64b8?method=getImage&version=38873&cstk=Xuv5iEhK" />

前向计算图

#### 隐层

- 线性计算

$$
```math

z1_{1} = x \cdot w1_{11} + b1_{1}

```
$$

$$
```math

z1_{2} = x \cdot w1_{12} + b1_{2}

```
$$

$$
```math

z1_{3} = x \cdot w1_{13} + b1_{3}

```
$$

矩阵形式：

$$
```math

\begin{aligned}
Z1 &=x \cdot 
\begin{pmatrix}
    w1_{11} & w1_{12} & w1_{13}
\end{pmatrix}
+
\begin{pmatrix}
    b1_{1} & b1_{2} & b1_{3}
\end{pmatrix}
 \\\\
&= X \cdot W1 + B1  
\end{aligned} \tag{1}

```
$$

- 激活函数

$$
```math

a1_{1} = Sigmoid(z1_{1})

```
$$

$$
```math

a1_{2} = Sigmoid(z1_{2})

```
$$

$$
```math

a1_{3} = Sigmoid(z1_{3})

```
$$

矩阵形式：

$$
```math

A1 = Sigmoid(Z1) \tag{2}

```
$$

#### 输出层

由于我们只想完成一个拟合任务，所以输出层只有一个神经元：
$$
```math

\begin{aligned}
Z2&=a1_{1}w2_{11}+a1_{2}w2_{21}+a1_{3}w2_{31}+b2_{1} \\\\
&= 
\begin{pmatrix}
a1_{1} & a1_{2} & a1_{3}
\end{pmatrix}
\begin{pmatrix}
w2_{11} \\\\ w2_{21} \\\\ w2_{31}
\end{pmatrix}
+b2_1 \\\\
&=A1 \cdot W2+B2
\end{aligned} \tag{3}

```
$$



#### 损失函数

均方误差（MSE）损失函数：

$$
```math
loss(w,b) = \frac{1}{2} (z2-y)^2 \tag{4}
```
$$

其中，$z2$是预测值，$y$是样本的标签值。

> 标准均方误差的公式：
> $$
> ```math
> loss(w,b) = \frac{1}{n}{\sum_{i=1}^n{[f(x_i)-y_i]^2}}
> ```
> $$
> 其实就是一批数据与单个数据的区别。

### 3.反向传播

接下来，咱们来推导一下反向传播的各个过程。看一下计算图，然后用链式求导法则反推。

#### 求损失函数对输出层的反向误差

根据公式4：

$$
```math

\frac{\partial loss}{\partial z2} = z2 - y \rightarrow dZ2 \tag{5}

```
<img src="https://note.youdao.com/yws/api/personal/file/WEB351e164ac44aa191f769d7e2f7b248c3?method=getImage&version=38931&cstk=Xuv5iEhK" />
$$

#### 求W2的梯度

根据公式3和W2的矩阵形状，把标量对矩阵的求导分解到矩阵中的每一元素：

$$
```math

\begin{aligned}
\frac{\partial loss}{\partial W2} &= 
\begin{pmatrix}
    \frac{\partial loss}{\partial z2}\frac{\partial z2}{\partial w2_{11}} \\\\
    \frac{\partial loss}{\partial z2}\frac{\partial z2}{\partial w2_{21}} \\\\
    \frac{\partial loss}{\partial z2}\frac{\partial z2}{\partial w2_{31}}
\end{pmatrix} \\\\
&=\begin{pmatrix}
    dZ2 \cdot a1_{1} \\\\
    dZ2 \cdot a1_{2} \\\\
    dZ2 \cdot a1_{3}
\end{pmatrix} \\\\
&=\begin{pmatrix}
    a1_{1} \\\\ a1_{2} \\\\ a1_{3}
\end{pmatrix} \cdot dZ2
=A1^{\top} \cdot dZ2 \rightarrow dW2
\end{aligned} \tag{6}

```
$$

<img src="https://note.youdao.com/yws/api/personal/file/WEB39679ef6b511fef16c697d98d22848c9?method=getImage&version=38941&cstk=Xuv5iEhK" />

<img src="https://note.youdao.com/yws/api/personal/file/WEB45615264655280a7d8a26ba6eb5cf936?method=getImage&version=38939&cstk=Xuv5iEhK" />



#### 求B2的梯度

$$
```math

\frac{\partial loss}{\partial B2}=dZ2 \rightarrow dB2 \tag{7}

```
$$

<img src="https://note.youdao.com/yws/api/personal/file/WEB0626d42235a24ca21f29825120f48251?method=getImage&version=38949&cstk=Xuv5iEhK" />
<img src="https://note.youdao.com/yws/api/personal/file/WEB59fb8621a9917cfbf5969dce16f210e0?method=getImage&version=38950&cstk=Xuv5iEhK" />

对于输出层来说，A就是它的输入，也就相当于是X。

#### 求损失函数对隐层的反向误差

下面的内容是也是深度神经网络的基础，请大家仔细阅读体会。

<img src="https://note.youdao.com/yws/api/personal/file/WEBb01908a2e7eb59b5c6f94afea3041dde?method=getImage&version=38902&cstk=Xuv5iEhK" />

正向计算和反向传播路径图

图中：

- 蓝色矩形表示数值或矩阵；
- 蓝色圆形表示计算单元；
- 蓝色的箭头表示正向计算过程；
- 红色的箭头表示反向计算过程。

如果想计算W1和B1的反向误差，必须先得到Z1的反向误差，再向上追溯，可以看到Z1->A1->Z2->Loss这条线，Z1->A1是一个激活函数的运算，比较特殊，所以我们先看Loss->Z->A1如何解决。

根据公式3和A1矩阵的形状：

$$
```math

\begin{aligned}
\frac{\partial loss}{\partial A1}&=
\begin{pmatrix}
    \frac{\partial loss}{\partial Z2}\frac{\partial Z2}{\partial a1_{11}} \\\\
    \frac{\partial loss}{\partial Z2}\frac{\partial Z2}{\partial a1_{12}} \\\\
    \frac{\partial loss}{\partial Z2}\frac{\partial Z2}{\partial a1_{13}}
\end{pmatrix} \\\\
&=
\begin{pmatrix}
    dZ2 \cdot w2_{11} \\\\
    dZ2 \cdot w2_{12} \\\\
    dZ2 \cdot w2_{13}
\end{pmatrix} \\\\
&=dZ2 \cdot
\begin{pmatrix}
    w2_{11} \\\\ w2_{21} \\\\ w2_{31}
\end{pmatrix}^{\top}=dZ2 \cdot W2^{\top}
\end{aligned} \tag{8}

```
$$

<img src="https://note.youdao.com/yws/api/personal/file/WEB23b3f322e82e90ef90a7017a14f47466?method=getImage&version=38965&cstk=Xuv5iEhK" />
<img src="https://note.youdao.com/yws/api/personal/file/WEB516ee57ab3be7c34c9604d33f9fe75b0?method=getImage&version=38964&cstk=Xuv5iEhK" />

现在来看激活函数的误差传播问题，由于公式2在计算时，并没有改变矩阵的形状，相当于做了一个矩阵内逐元素的计算，所以它的导数也应该是逐元素的计算，不改变误差矩阵的形状。根据Sigmoid激活函数的导数公式，有：

$$
```math

\frac{\partial A1}{\partial Z1}= Sigmoid'(A1) = A1 \odot (1-A1) \tag{9}

```
$$

<img src="https://note.youdao.com/yws/api/personal/file/WEB616012495d0f6d8b62fbdb4285dcc5c4?method=getImage&version=38972&cstk=Xuv5iEhK" />
<img src="https://note.youdao.com/yws/api/personal/file/WEB7e8af10d61bba1b023bf3ab74bda7799?method=getImage&version=38971&cstk=Xuv5iEhK" />

所以最后到达Z1的误差矩阵是：

$$
```math

\begin{aligned}
\frac{\partial loss}{\partial Z1}&=\frac{\partial loss}{\partial A1}\frac{\partial A1}{\partial Z1} \\\\
&=dZ2 \cdot W2^T \odot Sigmoid'(A1) \rightarrow dZ1
\end{aligned} \tag{10}

```
$$

有了dZ1后，再向前求W1和B1的误差，我们直接列在下面：

$$
```math

dW1=X^T \cdot dZ1 \tag{11}

```
$$
<img src="https://note.youdao.com/yws/api/personal/file/WEBf2258d44a05d6c5da0415192ff975b85?method=getImage&version=38977&cstk=Xuv5iEhK" />
$$
```math

dB1=dZ1 \tag{12}

```
$$

之后只需要根据计算的梯度来更新相应的权重，就可以保证模型的自我修正、拟合、收敛了。

什么是梯度

梯度是微积分中的基本概念，也是机器学习解优化问题经常使用的数学工具（梯度下降算法），虽然常说常听常见，但其细节、物理意义以及几何解释还是值得深挖一下，这些不清楚，梯度就成了“熟悉的陌生人”。

**导数是一元函数的变化率（斜率）**。导数也是函数，是函数的变化率与位置的关系。

如果是多元函数呢？则为**偏导数**。

**偏导数是多元函数“退化”成一元函数时的导数**，这里“退化”的意思是**固定其他变量的值，只保留一个变量**，依次保留每个变量，则N元函数有N个偏导数。

偏导数构成的向量为梯度；


<video id="video" controls="" preload="none" poster="http://om2bks7xs.bkt.clouddn.com/2017-08-26-Markdown-Advance-Video.jpg">
<source id="mp4" src="https://upos-sz-mirror08ct.bilivideo.com/upgcxcode/21/06/129160621/129160621-1-6.mp4?e=ig8euxZM2rNcNbRVhwdVhwdlhWdVhwdVhoNvNC8BqJIzNbfq9rVEuxTEnE8L5F6VnEsSTx0vkX8fqJeYTj_lta53NCM=&uipk=5&nbs=1&deadline=1680239970&gen=playurlv2&os=08ctbv&oi=17627301&trid=50273d77de8f42c98ac47a5d3e955451h&mid=0&platform=html5&upsig=263f5396f35ad5310f4012a9b4724bd7&uparams=e,uipk,nbs,deadline,gen,os,oi,trid,mid,platform&bvc=vod&nettype=0&bw=28581&logo=80000000" type="video/mp4">
</video>


刚才讲的只是Sigmoid函数、线性回归损失函数的梯度，其他的激活函数、损失函数各自都有自己的梯度，但反向传播的过程都是一样的。

链式法则

OpenAI
要做的是AGI（通用人工智能）

模型语言翻译师

let's think step by step
让我们想想一步一步来

dalle2的语言


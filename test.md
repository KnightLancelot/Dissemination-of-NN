
# 详解前向计算、反向传播

#### **开始了解概念之前，咱们先通过”[Tensorflow Playground](http://playground.tensorflow.org/)“，来可视化的体验一下简单神经网络的训练过程。**

## 概念介绍

机器学习并没有什么魔法，它其实就是一个找出过往数据的内在规律，并对未来新数据进行预测的过程。所有的机器学习，包括深度学习，都是找到从输入到输出的最佳拟合函数的过程。传统机器学习可能用的是一些从统计学中传承的方法，而深度学习作为机器学习的一个新领域，则是从生物神经网络原理中得到的灵感，用网状结构逐步调整各神经元权重的方法来拟合函数。


神经网络之所以能工作，靠的就是“**前向计算**”、“**梯度下降**和**反向传播**”。而我们在学习这部分知识时，却总是发现相关内容往往充斥着各种数学公式，具体的推导过程又往往语焉不详，经常看的人云里雾里的不知所云。
今天咱们就通过一个例子，来详解一下神经网络在“**前向计算**”、“**梯度下降**和**反向传播**”时都做了些什么。


#### **梯度下降**
> 梯度是微积分中的基本概念，也是机器学习中解优化问题时经常使用的数学工具（梯度下降算法）。虽然我们对梯度可能是常说常听常见，但其细节、物理意义以及几何解释还是值得深挖一下的，这些如果不清楚，梯度就成了“熟悉的陌生人”。
> 
> 梯度的本意是一个向量（矢量），表示某一函数在该点处的方向导数沿着该方向取得的最大值，即函数在该点处**沿着该方向（此梯度的方向）变化最快，变化率最大**（为该梯度的模）。
> 
> 所谓导数，就是用来分析函数“变化率”的一种度量。针对函数中的某个特定点x0，该点的导数就是x0点的“瞬间斜率”，也即（一元函数的）切线（变化率）**斜率**。
> 
> 这个斜率越大，则表示其上升趋势越强劲。当这个斜率为0时，就达到了这个函数的“强弩之末”，即达到了极值点。
> 在单变量的实值函数中，**梯度可简单理解为就是导数**，或者说对于一个线性函数而言，**梯度就是曲线在某点的斜率**。
> 
> 因为，梯度求的是函数变化率最大的方向，所以，梯度下降算法通过计算可微函数的梯度并**沿梯度的相反方向移动**，指导搜索局部/全局最小值，以最小化函数的值。
> 
> 多元函数时，求出的**导数**则为**偏导数**。
> 偏导数是多元函数“退化”成一元函数时的导数，这里“退化”的意思是**固定其他变量的值，只保留一个变量**，依次保留每个变量，则**N元函数有N个偏导数**。
> 
> [偏导数和梯度向量的三维可视化视频](https://github.com/KnightLancelot/Dissemination-of-NN/blob/main/files/%E5%81%8F%E5%AF%BC%E6%95%B0%E5%92%8C%E6%A2%AF%E5%BA%A6%E5%90%91%E9%87%8F%E7%9A%84%E4%B8%89%E7%BB%B4%E5%8F%AF%E8%A7%86%E5%8C%96%EF%BC%88%E5%B0%8F%EF%BC%89.mp4?raw=true)



## **What I cannot create, I do not understand.**
## **如果我不能亲手搭建起来一个东西，那么我就不能理解它。 -- 美国物理学家理查德·费曼**

### 1.定义神经网络结构

本例的目的是要讲清楚，神经网络在完成曲线拟合时，是如果”前向计算“、”反向传播“的。

<img src="https://github.com/KnightLancelot/Dissemination-of-NN/blob/main/files/complex_result_3n.png?raw=true" />
上图为曲线拟合示意图

根据万能近似定理的要求，我们定义一个两层的神经网络，输入层不算，一个隐藏层，含3个神经元，一个输出层。下图显示了此次用到的神经网络结构。

<img src="https://github.com/KnightLancelot/Dissemination-of-NN/blob/main/files/nn.png?raw=true" />
上图为单入单出的双层神经网络

#### 输入层

输入层就是一个标量x值（如果是成批输入，则是一个矢量或者矩阵）。

```math
X = (x)
```

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

<img src="https://github.com/KnightLancelot/Dissemination-of-NN/blob/main/files/forward.png?raw=true" />
上图为前向计算图

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

下面的内容是深度神经网络的基础，请大家仔细阅读体会。

<img src="https://github.com/KnightLancelot/Dissemination-of-NN/blob/main/files/backward.png?raw=true" />
上图为正向计算和反向传播路径图

图中：

- 蓝色矩形表示数值或矩阵；
- 蓝色圆形表示计算单元；
- 蓝色的箭头表示正向计算过程；
- 红色的箭头表示反向计算过程。


#### 求损失函数对输出层的反向误差

根据公式4：

$$
```math

\frac{\partial loss}{\partial z2} = z2 - y \rightarrow dZ2 \tag{5}

```
$$
<img src="https://github.com/KnightLancelot/Dissemination-of-NN/blob/main/files/loss%E5%AF%B9Z2%E7%9A%84%E6%B1%82%E5%AF%BC.jpg?raw=true" />

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

<img src="https://github.com/KnightLancelot/Dissemination-of-NN/blob/main/files/loss%E5%AF%B9W2%E7%9A%84%E9%93%BE%E5%BC%8F%E6%B1%82%E5%AF%BC.jpg?raw=true" />
<img src="https://github.com/KnightLancelot/Dissemination-of-NN/blob/main/files/Z2%E5%AF%B9W2%E7%9A%84%E6%B1%82%E5%AF%BC.jpg?raw=true" />


#### 求B2的梯度

$$
```math

\frac{\partial loss}{\partial B2}=dZ2 \rightarrow dB2 \tag{7}

```
$$

<img src="https://github.com/KnightLancelot/Dissemination-of-NN/blob/main/files/loss%E5%AF%B9B2%E7%9A%84%E9%93%BE%E5%BC%8F%E6%B1%82%E5%AF%BC.jpg?raw=true" />
<img src="https://github.com/KnightLancelot/Dissemination-of-NN/blob/main/files/Z2%E5%AF%B9B2%E7%9A%84%E6%B1%82%E5%AF%BC.jpg?raw=true" />


#### 求损失函数对隐层的反向误差

根据公式3和A1矩阵的形状：

$$
```math

\begin{aligned}
\frac{\partial loss}{\partial A1}&=
\begin{pmatrix}
    \frac{\partial loss}{\partial Z2}\frac{\partial Z2}{\partial a1_{11}}
    &
    \frac{\partial loss}{\partial Z2}\frac{\partial Z2}{\partial a1_{12}}
    &
    \frac{\partial loss}{\partial Z2}\frac{\partial Z2}{\partial a1_{13}}
\end{pmatrix} \\\\
&=
\begin{pmatrix}
dZ2 \cdot w2_{11} & dZ2 \cdot w2_{12} & dZ2 \cdot w2_{13}
\end{pmatrix} \\\\
&=dZ2 \cdot
\begin{pmatrix}
    w2_{11} & w2_{21} & w2_{31}
\end{pmatrix} \\\\
&=dZ2 \cdot
\begin{pmatrix}
    w2_{11} \\\\ w2_{21} \\\\ w2_{31}
\end{pmatrix}^{\top}=dZ2 \cdot W2^{\top}
\end{aligned} \tag{8}

```
$$

<img src="https://github.com/KnightLancelot/Dissemination-of-NN/blob/main/files/loss%E5%AF%B9A1%E7%9A%84%E9%93%BE%E5%BC%8F%E6%B1%82%E5%AF%BC.jpg?raw=true" />
<img src="https://github.com/KnightLancelot/Dissemination-of-NN/blob/main/files/Z2%E5%AF%B9A1%E7%9A%84%E6%B1%82%E5%AF%BC.jpg?raw=true" />

现在来看激活函数的误差传播问题，由于公式2在计算时，并没有改变矩阵的形状，相当于做了一个矩阵内逐元素的计算，所以它的导数也应该是逐元素的计算，不改变误差矩阵的形状。根据Sigmoid激活函数的导数公式，有：

$$
```math

\frac{\partial A1}{\partial Z1}= Sigmoid'(A1) = A1 \odot (1-A1) \tag{9}

```
$$

<img src="https://github.com/KnightLancelot/Dissemination-of-NN/blob/main/files/sigmoid%E5%87%BD%E6%95%B0%E6%B1%82%E5%AF%BC.jpg?raw=true" />
<img src="https://github.com/KnightLancelot/Dissemination-of-NN/blob/main/files/e%E7%9A%84%E8%B4%9Fx%E6%96%B9%E6%B1%82%E5%AF%BC.jpg?raw=true" />

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
<img src="https://github.com/KnightLancelot/Dissemination-of-NN/blob/main/files/loss%E5%AF%B9W1%E7%9A%84%E6%B1%82%E5%AF%BC.jpg?raw=true" />
$$
```math

dB1=dZ1 \tag{12}

```
$$

之后只需要根据计算的梯度来更新相应的权重、偏置，经过多轮训练，模型就会逐步的自我修正、拟合、收敛了。


刚才讲的只是Sigmoid函数、线性回归损失函数的梯度，其他的激活函数、损失函数各自都有自己的梯度，但反向传播的过程都是一样的。


这里要明确一点，就像分子构成细胞，细胞构成组织、器官一样，刚才讲解的这个案例，充其量只能类比为最简单的细胞，仅用来说明深度学习（神经网络）工作的基本原理。
现代的神经网络模型，就是在分子（"前向计算"、“反向传播”……）的基础上，构成了多种多样的细胞（CNN、RNN、ResNet、R-CNN……），进而构成了各种组织、器官（GAN、UNet、YOLO、Transformer Encoder、Transformer Decoder……），最终构成了形态各异的生物（Bert、GPT、DETR、MAE、CLIP、DALL·E……）。

## **观察与思考**
接下来，我再蹭一下风头正劲的LLM（大语言模型）的热度。
就在上个月，OpenAI发布了GPT-4。
正如[GPT-4的论文](https://arxiv.org/pdf/2303.08774.pdf)（技术报告）中提到的，GPT-4已经可以通过律师资格证考试，而且还超过了90%的考生，刷榜了大多数偏文科考试。
还有人用GPT-4测试，通过了Google、微软等各大公司的面试。
GPT-4还能帮人做游戏、3D建模、做投资等等，总之就是强的令人发指！

紧接着，OpenAI又发布了一个[LLM会对劳动力市场有哪些影响的报告](https://arxiv.org/pdf/2303.10130.pdf)。
文中指出，80%的美国劳动力平时工作中有10%的任务，会受到LLM的影响。另有19%的劳动力，他们50%的工作会受到LLM的影响。
高学历的人似乎更容易被AI所取代。此外，更高门槛的工作、更高收入的工作往往也更容易被AI所取代。不过文章同时也指出，那些需要科学能力和批判思维能力的工作暴露率（高风险率）较低。
专家和模型各选了一批受影响最严重的职业，其中包括：数学家、报税员、量化金融分析师、作家、网络和数字界面设计师，还包括新闻记者、法务、行政、通讯员、区块链工程师、译员、公关专家、调研员等职业。
影响最小的职业，都是那些体力工作者：农机操作员、运动员、食品服务业、林业和伐木业、社会救助和食品制造业……
职业类型之外，研究团队也按照行业分析了LLM的影响。结果显示，包括数据处理托管、出版行业和安全商品合约在内的行业，最可能被颠覆。

> 所谓“暴露率”，其判断标准是通过访问GPT或GPT驱动的系统是否会使人类执行一项任务所需的时间减少50%以上。“虽然暴露并不一定表明他们的工作任务可以通过GPT技术完全自动化，但它们将帮助这些人节省大量完成任务的时间。”


其实，就我个人对这些LLM或LLM提供能力的工具（微软的Copilot、GitHub的Copilot），也有一些观察与思考。
目前LLM能提供的能力，往往体现在拉高了所有人能力阈值的方面。
即在某些非专长的工作中，一个专业技能不足的人也可以通过LLM的协助，比较漂亮的完成一些难度有限的工作。
而对于一个资深的专业人士来说，使用LLM获得的专业能力提升就比较有限了——LLM往往就只能帮助解决一些技术含量较低、重复性较高的工作了。

所以，对现阶段的LLM既没必要过度恐慌，也不能盲目轻视。
不恐慌：从专业领域来看，LLM目前还只能解决一些痒点的问题。
不轻视：但是随着不断地迭代，LLM迟早有一天会变成一个真正多模态的AGI，成为一个有能力解决痛点问题的存在。

最近，MetaAI发布的Toolformer模型已经可以利用工具了，可以去调用各种各样(日历、浏览器、计算器……)的API。
OpenAI也迅速跟进推出了ChatGPT plugin，来连接成百上千的API。这就让大语言模型变成了一个交互的入口，整合起了已有的各种工具以及互联网，不仅提升了大语言模型的各种能力，还能通过互联网及时的更新模型的知识库，从而开启了无限的可能。

#### **说点闲话**
就我们个人来说，其实没别的选择——打不过就加入，紧跟未来的科技发展方向，拥抱AI。

这个拥抱AI，既可以理解为从事AI相关技术的研究、工业落地等工作；也可以理解为将AI作为一个黑盒，对其的交互方式进行研究、使用。
总之，就是不要离AI太远。

有鉴于现在LLM的使也有一定的门槛，我这里就通过几个例子，来简单的给大家介绍一下。

1. 研究社区中有人提出了“Large Language Models are Zero-Shot Reasoners“，指出只需要在LLM给出答复之前，增加一句“Let's think step by step”，就很能立刻让模型在两个比较难的数学问题数据集上大比例涨点！
<img src="https://github.com/KnightLancelot/Dissemination-of-NN/blob/main/files/step_by_step0.png?raw=true" />
研究人员进而还提出了“Chain of Thought”(CoT)理论、“AI也要求鼓励”的观点。
<img src="https://github.com/KnightLancelot/Dissemination-of-NN/blob/main/files/step_by_step1.png?raw=true" />

1. 还是研究社区中最先有人发现，可以通过给ChatGPT“催眠”来突破模型的限制。
<img src="https://github.com/KnightLancelot/Dissemination-of-NN/blob/main/files/DAN%E7%AA%81%E7%A0%B4ChatGPT%E7%9A%84%E9%99%90%E5%88%B6.png?raw=true"/>
之后OpenAI才开始跟进，在GPT-4中提供了一个新的特性“System messages”，可以让GPT-4按照用户要求的风格，与用户交互。参见：[Steerability: Socratic tutor](https://openai.com/research/gpt-4)。


其实这些LLM本身就是个黑盒，包括设计、训练这些LLM的人也不知道模型内部到底在干什么。
以至于GPT3的论文中都指出，由于模型太大、数据量太多，他们（OpenAI）也不知道，模型到底是记住了那些数据，还是真的学到了客观规律。
综上所述，想用好这些LM也是需要一定的研究、学习门槛的，而这些知识也可能在未来，成为个人的竞争优势。

### 尾巴
最后，让我们用一个小故事来结束这篇文章。
ChatGPT版的回形针问题
<img src="http://res.cloudinary.com/lesswrong-2-0/image/upload/v1669986602/mirroredImages/RYcoJdvmoBbi5Nax7/ledcpw7odbjittuwmysn.jpg"/>
<img src="http://res.cloudinary.com/lesswrong-2-0/image/upload/v1669986602/mirroredImages/RYcoJdvmoBbi5Nax7/rdfcb6yfheernbv93ksg.jpg"/>

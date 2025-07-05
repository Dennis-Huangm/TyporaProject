# <center>Deep Dive into Transformer Architecture</center>

## 1  Architectural Foundations

### 1.1  The Architectural of GPT-2  

​	我们以GPT-2的**模型架构**为切入，分析整个Transformer Block的**结构**及其**内在机制**。GPT-2的架构是在GPT-1的基础上改进的，而GPT-1的模型架构则是拿掉了Multi-Head Cross Attention （多头交叉注意力），只保留了Masked Multi-Head Self-Attention的**Transformer的解码器**。GPT-2的模型架构在GPT-1的基础上做了如下改进：

- Layer normalization被移动到每一个sub-block（两个子层：**解码器自注意力**与**基于位置的前馈神经网络**）的**输入**位置，类似于一个**预激活**的残差网络。同时在**最后的**自注意力块后添加一个额外的layer normalization。
- 采用一种改进的初始化方法，该方法考虑了残差路径与模型深度的累积。在初始化阶段使用缩放因子$\frac{1}{\sqrt{N}}$对residual layer的权重进行缩放操作，其中$N$为residual layer的数量（深度）。
- 字典大小设置为50257；无监督预训练可看到的上下文的 context 由512扩展为1024；Batch Size 大小调整为512。

<img src="./assets/未命名绘图.drawio (2)-1751705732359-2.png" alt="未命名绘图.drawio (2)" style="zoom:40%;" />

<center style="color:#C0C0C0">图1 GPT-2的Transformer Block</center>

### 1.2  Transformer Block 构成分析

​	假设我们的输入为一个$X\in\mathbb{R}^{n\times d}, \quad x_i^{\!\top}\in\mathbb{R}^{1\times d}\;(i=1,\dots,n)$的矩阵，其中每一行为一个token的表征向量，长度为$d$（相当于经过了embedding与position encoding操作），接下来它将经过layer normalization、Multi-Head Self-Attention、Position‑wise Feed‑Forward Networks等计算操作，我们一个一个来分析。

#### 1.2.1  Layer Normalization

​	层归一化按**行**（即每个 token）归一化，对同一token的全部特征做**零均值**、**单位方差**处理，公式如下（对于第$i$个token有）：
$$
\mu_i=\frac1d\mathbf1^{\!\top}x_i , \quad  
\sigma_i\;=\;\bigl(\tfrac1d\|x_i-\mu_i\mathbf1\|_2^2\bigr)^{1/2},\qquad  
\hat x_i=\frac{x_i-\mu_i\mathbf1}{\sigma_i+\varepsilon}.
$$
堆叠得到：
$$
\hat X=\bigl(I_n\otimes\mathbf1_d^{\!\top}\bigr)^{-1}(X-\mu\mathbf1_d^{\!\top})\oslash\sigma
$$
​	

​	以下为Layer Normalization的优点：**只依赖行内统计**（不需存储/维护全局运行均值与方差；只对最后一维做并行归约），与 batch size 无关，因此测试与训练过程完全一致。同时，LN 能减小层输入**尺度漂移**（internal covariate shift），在注意力与残差结构叠加时，能保持梯度在深层网络中有效传播，加速收敛。


















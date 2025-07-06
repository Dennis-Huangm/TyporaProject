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
\mu_i=\frac1d\mathbf1^{\!\top}x_i \in \mathbb{R} , \quad  
\sigma_i\;=\;\bigl(\tfrac1d\|x_i-\mu_i\mathbf1\|_2^2\bigr)^{1/2} \in \mathbb{R},\qquad  
\hat x_i=\frac{x_i-\mu_i\mathbf1}{\sigma_i+\varepsilon}\in \mathbb{R}^{d \times 1}
$$
堆叠得到：
$$
\hat{X}=(X-\mu\mathbf{1}_d^\top)\oslash(\sigma\mathbf{1}_d^\top+\varepsilon)
$$
​	其中，$\mu\in \mathbb{R}^{n \times 1}$，$\sigma \in \mathbb{R}^{n \times 1}$均为堆叠而成的**向量**；；$\oslash$为**Hadamard除**（矩阵逐元素相除）；$\mathbf1_d \in \mathbb{R}^{d \times 1}$为全1列向量；$\varepsilon \in \mathbb{R}^{n \times d}$，用于维持数值稳定。最后加上仿射变换的结果为：
$$
\mathrm{LN}(X)=\hat{X}\odot\gamma^\top+\beta^\top,\quad\gamma,\beta\in\mathbb{R}^d
$$
​	以下为Layer Normalization的优点：**只依赖行内统计**（不需存储/维护全局运行均值与方差；只对最后一维做并行归约），与 batch size 无关，因此测试与训练过程完全一致。同时，LN 能减小层输入**尺度漂移**（internal covariate shift），在注意力与残差结构叠加时，能保持梯度在深f网络中有效传播，加速收敛。
​	Transformer 在**小批量甚至序列长度为 1 的自回归推断**场景中尤为常见，LN**仅涉及当前 token 向量本身**，推断时与训练时的分布完全对齐，无需像 BatchNorm 那样维护滑动均值，也不会出现 batch 幅度微抖动导致的生成质量劣化问题。

​	从几何视角来看，LN的操作是将所有的**token投影到超球面**中：

1. **平移**：$x\mapsto x-\mu_{i}$ —— 消除径向偏移；
2. **径向缩放**：除以 $\sigma$ —— 投影到半径 1 的球面；
3. **各向异性伸缩**：$\odot\gamma$ —— 把球面拉成椭球，提供可学习尺度。

因此 LN 把每个 token 的**向量表示**都 **压到同一“球壳”**（或椭球壳）上；后续注意力仅关心 **方向信息**，点积 $\langle q_i,k_j\rangle$ 规模始终 $\mathcal{O}(1)$，softmax区间稳定。

#### 1.2.2  Multi-Head Self-Attention

​	首先考虑标准的缩放点积注意力机制，对于任一头(head) $h$，可以得到投影矩阵:
$$
Q_{h}=XW_{Q}^{(h)},K_{h}=XW_{K}^{(h)},V_{h}=XW_{V}^{(h)},\quad W_{Q,K,V}^{(h)}\in\mathbb{R}^{d\times d_h}
$$
​	其中 $d_h$ 为**query/key空间**的维度，一般远小于**嵌入空间**的维度 $d$ 。投影矩阵的作用是将**嵌入(Embedding)空间**中的token**映射**到**较小**的**查询、键、值空间**中的某个方向。当键与查询的方向相对齐时，就能认为他们相匹配（高度对齐）。

<img src="./assets/未命名绘图.drawio-1751787896040-2.png" alt="未命名绘图.drawio" style="zoom:40%;" />

<center style="color:#C0C0C0">图2 查询向量与键向量的几何关系示意图，其中左图为嵌入空间(Embedding)，右图则为查询/键(query/key)空间</center>

​	未加掩码时的**单头注意力权重**为：
$$
S_h=\frac{Q_hK_h^\top}{\sqrt{h}}\in\mathbb{R}^{n\times n},\quad A_h=\mathrm{softmax}(S_h)
$$
​	$Q_hK_h^\top$结果的**每个元素**都可以看作一对**键—查询对**之间的点积，根据点积的概念，可以容易看出**值越大**说明键与查询越**对齐**。同时，为了维持数值稳定性，所有点积的结果都会除以**键—查询空间维度的平方根**。
​	由于 softmax 逐行作用，$A_h$的每一行都是一组**概率分布** (行随机矩阵)，每个元素都是一个注意力权重，表示一对键与查询向量之间的相关度。

​	GPT的本质还是一个**自回归的语言模型**，在预测阶段，其输出序列的词元是逐个生成的，因此同样需要**掩码**操作（这也是为什么它采用的是解码器，而**BERT**作为**双向**编码器无需掩码操作），即需要保证第 $i$ 个 token 不能看到序列中位置 $j>i$ 的信息，令：
$$
M_{ij}=
\begin{cases}
0, & j\leq i \\
-\infty, & j>i & 
\end{cases}
$$
​	记该“上三角”$M$矩阵为Masked Attention权重：
$$
A_h=\mathrm{softmax}(S_h+\mathbf{M})
$$
​	对于行 $i$ 只把允许的列保持原值，其余置 $-\infty$，softmax 后相当于把不合法位置的概率压到 0。

​	在实践中，当给定相同的香询、键和值的集合时，我们希望模型可以基于**相同的注意力机制**学习到**不同的行为**，然后将不同的行为作为知识组合起来，捕获序列内各种范围的依赖关系(例如，短距离依赖和长距离依赖关系)。

​	为此，我们可以用独立学习得到的 $H$ 组不同的**线性投影(MLP**)来变换查询、键和值。然后，这$h$组变换后的査询、键和值将并行地送到**注意力汇聚**中。最后，将这 $H$ 个注意力汇聚的输出**拼接在一起**，并且通过另一个可以学习的线性投影进行变换，以产生最终输出。这种设计即为**多头注意力机制**，对于 $h$ 个注意力汇聚输出，每一个注意力汇聚都被称作一个**头**(head)。

<img src="./assets/未命名绘图.drawio-1751789084903-6.png" alt="未命名绘图.drawio" style="zoom:50%;" />

​	对于每个注意力头 $\mathbf{h}_i(i=1,\dots,H)$：
$$
\mathbf{h}=A_hV_h\in\mathbb{R}^{n\times d_h}
$$
​	经拼接后再投影回 $d$（即嵌入空间）：
$$
MHSA(X)=[\mathbf{h}_1,\dots,\mathbf{h}_h]W_O+b_O, \quad W_O\in \mathbb{R}^{Hd_h\times d}
$$
​	基于这种设计，每个头都可能会关注输入的不同部分，可以表示比简单加权平均值更复杂的函数。

#### 1.2.3 Position-wise Feed-Forward Network

​	基于位置的前馈网络对**序列中的所有位置的表示进行变换**时，使用的是**同一个多层感知机(MLP)**，这就是称前馈网络是基于位置的原因：
$$
FFN(x)=\sigma(XW_1+b_1)W_2+b_2
$$
​	其中$W_1\in\mathbb{R}^{d \times d_{ff}}$，相当于升维的操作；$\sigma$为激活函数，如Relu；$W_2\in\mathbb{R}^{d_{ff} \times d }$，相当于降维的操作，回到原通道数；$b_{1,2}$为偏置项。

​	这个操作将自注意力产生的 **方向特征** 转换成 **坐标系内的高阶混合特征**，补足网络的非线性表达力。

### 1.3  Transformer的有效性分析

​	从**秩**的视角审视纯注意力网络（Self-Attention Network, SAN），发现：在没有跳跃连接（skip connections）和前馈网络（MLP）的情形下，随着层数加深，其**输出矩阵会以双指数速度退化到秩 1**——即所有 **token 最终“趋于同质”** 。

​	Skip Connection通过**允许信息绕过某些Self-Attention层**，从而在路径分解中引入了大量**短路径** 。最极端的情况是一条长度为0的路径，它直接将原始输入传递到输出，完整保留了输入的秩。这些短路径不会经历Deep Layer导致的严重秩坍塌，因此它们的存在有效地阻止了整个网络输出的退化，这揭示了跳跃连接在Transformer中一个此前未被充分认识的关键作用：**防止秩坍塌**。

​	MLP块作为**非线性变换**，可以增**加其输入矩阵的秩**，与Self-Attention层的降秩进行博弈。MLP的能力可以通过其**Lipschitz** constant来衡量，**Lipschitz常数越大的MLP，其提升秩的能力越强**，从而能更有效地减缓秩坍塌的速度。我们将在后面的篇章对上述理论进行详细证明。

## 2   Spectral Properties of Attention

​	我们先来解释一下什么是谱范数：**谱范数（spectral norm）**是矩阵 $A\in\mathbb{R}^{m\times n}$在 $\ell_2$ 意义下的算子范数——也就是把它看成线性变换 $x\mapsto Ax$ 时对向量欧氏长度的“最大放大倍数”：
$$
\|A\|_2=\max_{x\neq0}\frac{\|Ax\|_2}{\|x\|_2}=\max_{\|x\|_2=1}\|Ax\|_2.
$$
​	如何理解呢？我们可以设想一个这样的场景：

<img src="./assets/output.png" alt="output" style="zoom:36%;" />

- 将$A$看作为一个**线性变换**，它将向量 $x$ 拉伸或压缩为 $Ax$；
- 把**单位球**$\|x\|_2=1$看成输入空间的“所有方向”，对其进行线性变换后会得到一个**椭球**（或更高维的超椭球）：$E=\{Ax:\|x\|_2=1\}$；
- “最大”指椭球的最长“半径”，**椭圆最远离原点的那一点**到原点的距离，对应的那条半径的长度就是谱范数；对应的方向叫 **主奇异向量**。










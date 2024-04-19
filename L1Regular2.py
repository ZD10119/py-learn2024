# 在一个简单的线性回归模型中加入 L1 正则化
# 假设我们有一个简单的线性模型，并且我们希望在损失函数中加入 L1 正则化

# 这段代码展示了如何在一个简单的线性回归模型的损失函数中加入 L1 正则化项。
# l1_lambda 是正则化强度，model.parameters() 是模型中所有的权重参数。

import torch
import torch.nn as nn

# 创建数据和模型
x = torch.randn(10, 3)
y = torch.randn(10, 1)
model = nn.Linear(3, 1)

# 前向传播
pred = model(x)

# 计算 MSE 损失
mse_loss = nn.MSELoss()(pred, y)

# 计算 L1 正则化损失
l1_lambda = 0.05  # 正则化系数
l1_norm = sum(p.abs().sum() for p in model.parameters())

# 总损失
total_loss = mse_loss + l1_lambda * l1_norm

print("Total loss:", total_loss)

# 表达式 model = nn.Linear(3, 1) 创建了一个线性层，这是神经网络中最基本的层之一，通常被称为全连接层或密集层。
'''### `nn.Linear`
- `nn` 是 PyTorch 中 `torch.nn` 的缩写，它是一个包含各种神经网络模块的库，如层、激活函数等。
- `Linear` 指的是线性模块，用于创建一个线性函数，即 `y = Wx + b`，其中 `W` 是权重矩阵，`b` 是偏置向量。

### 参数：(3, 1)
- 这里的参数 `(3, 1)` 定义了这个线性层的输入和输出特征数量。
- `3` 表示输入特征的数量。这意味着每个输入向量应该有3个元素。
- `1` 表示输出特征的数量。这意味着对于每个输入向量，线性层将输出一个单一的数值。

### 功能
- 当你将一个包含三个元素的张量（或更高维度的包含多个这样的张量的批量数据）传递给这个模型时，它会应用上面描述的线性转换。
- 在内部，`nn.Linear` 会随机初始化 `W` 和 `b`（除非另有指定），并在训练过程中通过反向传播算法学习这些参数的最佳值。

### 示例
如果你有一个特征向量 `[x1, x2, x3]`，这个模型会计算：y = W1*x1 + W2*x2 + W3*x3 + b
其中 `W1`, `W2`, `W3` 是权重，`b` 是偏置。

这个模型可以用于各种机器学习任务，包括回归和分类，取决于它后面是否接有其他层和如何处理输出。这个简单的层是构建更复杂神经网络模型的基础。'''

# 在使用 PyTorch 的 `nn.Linear` 模块时，权重矩阵 \( W \) 和偏置向量 \( b \) 是自动创建并进行初始化的，不需要手动输入这些值。
'''当创建一个 `nn.Linear` 层时，PyTorch 会按照预设的方法初始化这些参数，你可以根据需要进行调整。

下面是一些关于这些参数和它们初始化的详细信息：

### 自动初始化
- **权重 (W)** 和 **偏置 (b)** 都会被自动初始化。默认情况下，权重通过 Kaiming 初始化（也称为 He 初始化），偏置通常初始化为零。
- 这种初始化方式是为了帮助模型在训练初期保持稳定，尤其是在使用激活函数如 ReLU 时。

### 自定义初始化
如果你想要自定义这些参数的初始化方式，PyTorch 也提供了相应的接口。这里有一些例子说明如何进行自定义初始化：

#### 示例：自定义权重和偏置的初始化

import torch
import torch.nn as nn
# 创建一个线性层
model = nn.Linear(3, 1)
# 自定义权重初始化
torch.nn.init.normal_(model.weight, mean=0.0, std=1.0)
# 自定义偏置初始化
torch.nn.init.constant_(model.bias, 0.5)

在这个示例中：
- 我们使用了 `normal_` 方法来将权重初始化为均值为 0，标准差为 1 的正态分布。
- 使用 `constant_` 方法将偏置初始化为常数 0.5。

### 训练过程中的参数更新
- 在模型训练过程中，这些参数（权重和偏置）会根据损失函数和所选优化器自动更新。
- 你不需要手动改变这些参数，除非你正在进行一些特殊的操作或实验。

通过这种设计，PyTorch 使得构建和训练神经网络变得更加简单和直观，而无需担心底层的许多实现细节。这样可以让你更专注于模型的设计和训练过程。'''

# 表达式 l1_norm = sum(p.abs().sum() for p in model.parameters()) 用于计算模型所有参数的 L1 正则化项，也就是参数的绝对值之和。
'''1. **`model.parameters()`**:
   - 这是一个生成器，返回模型中所有的参数（通常包括权重和偏置）。
   - 这些参数是 `torch.Tensor` 对象。

2. **`p.abs()`**:
   - 对于每个参数 `p`，`p.abs()` 计算其元素的绝对值。
   - 这意味着如果参数中有负值，它们会被转换成正值。

3. **`p.abs().sum()`**:
   - `sum()` 函数将 `p.abs()` 计算得到的所有绝对值相加。
   - 这一步为每个参数张量计算了 L1 范数（即绝对值之和）。

4. **`sum(p.abs().sum() for p in model.parameters())`**:
   - 这个外部的 `sum()` 函数将所有单独参数的 L1 范数相加，得到整个模型的 L1 正则化值。
   - 这是通过列表推导式完成的，对模型中的每个参数应用相同的绝对值和求和操作。'''
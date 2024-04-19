# 使用优化器更新带有 L1 正则化的模型参数
# 这段代码创建了一个优化器，并在训练循环中使用 L1 正则化更新模型参数。每一步都会计算总损失，然后进行反向传播和参数更新。

import torch
import torch.nn as nn
import torch.optim as optim

# 创建数据和模型
x = torch.randn(10, 3)
y = torch.randn(10, 1)
model = nn.Linear(3, 1)
optimizer = optim.SGD(model.parameters(), lr=0.01)

l1_lambda = 0.05  # 正则化系数

# 训练步骤
for i in range(100):
    optimizer.zero_grad()
    pred = model(x)
    mse_loss = nn.MSELoss()(pred, y)
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    total_loss = mse_loss + l1_lambda * l1_norm
    total_loss.backward()
    optimizer.step()

print("Finished training with L1 regularization")
print("Total loss:", total_loss)

# 表达式 optimizer = optim.SGD(model.parameters(), lr=0.01) 创建了一个优化器对象，用于更新你的模型 model 的参数，以此优化（即最小化）模型的损失函数。
# 这个优化器使用了随机梯度下降（Stochastic Gradient Descent，简称 SGD）方法。
'''### 参数：`model.parameters(), lr=0.01`
- `model.parameters()` 是传递给优化器的参数，指定了哪些参数需要在训练过程中被更新。这通常包括模型的所有可训练参数（如权重和偏置）。
- `lr=0.01` 是学习率，这是一个关键的超参数，控制着参数更新的步长。学习率的大小直接影响到训练过程的速度和质量。太大的学习率可能导致训练过程不稳定，而太小的学习率则会使训练速度过慢。

### 功能和作用
- **优化器的目的** 是通过逐步调整模型参数来减少模型预测和实际数据之间的误差（即损失）。
- 在每个训练步骤中，优化器会根据模型参数对损失函数的梯度来更新这些参数。具体来说，它会按照梯度的反方向，以指定的学习率移动参数值，从而试图达到损失的局部最小。'''

# "训练步骤"这段代码是一个典型的 PyTorch 训练循环，用于训练一个神经网络模型。
# 这个循环会运行 100 次迭代，每次迭代中执行一系列步骤来更新模型的参数，以最小化定义的总损失。下面详细解读这段代码的每一部分：

'''### 循环结构
- `for i in range(100):` 表示循环将执行 100 次。这里的 `100` 是迭代次数，即训练过程中总的数据处理次数。
### 清除梯度
- `optimizer.zero_grad()` 用于清除（重置）之前的梯度计算。在 PyTorch 中，梯度是累加的，因此如果不清零，前一次迭代的梯度会影响当前的梯度计算。
### 计算模型预测
- `pred = model(x)` 使用当前模型 `model` 对输入数据 `x` 进行预测，得到预测结果 `pred`。
### 计算均方误差损失
- `mse_loss = nn.MSELoss()(pred, y)` 计算预测值 `pred` 和真实值 `y` 之间的均方误差（Mean Squared Error, MSE）。这是评估回归任务性能的常用损失函数。
### 计算 L1 正则化损失
- `l1_norm = sum(p.abs().sum() for p in model.parameters())` 计算模型所有参数的 L1 范数，即所有参数绝对值的总和。这有助于模型的泛化，防止过拟合。
### 计算总损失
- `total_loss = mse_loss + l1_lambda * l1_norm` 计算总损失，包括 MSE 损失和 L1 正则化损失的加权和。这里的 `l1_lambda` 是一个系数，用于调整 L1 正则化在总损失中的影响力度。
### 反向传播
- `total_loss.backward()` 执行反向传播，自动计算所有可训练参数（模型权重）的梯度。
### 参数更新
- `optimizer.step()` 根据优化器定义的方法（此例中为 SGD）和梯度更新模型的参数。

总结来说，这段代码通过结合 MSE 损失和 L1 正则化，训练一个模型来同时优化预测精度和模型复杂度，从而可能获得更好的泛化性能。通过 100 次迭代来逐步调整模型参数，以达到更低的总损失。'''






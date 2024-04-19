# 使用 PyTorch 演示了在神经网络中如何使用三种不同的规范化技术：
# 批量规范化（Batch Normalization）、层规范化（Layer Normalization）和组规范化（Group Normalization）

# 生成数据部分
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

np.random.seed(42)
X = np.random.rand(1000, 10)
y = (X.sum(axis=1) > 5).astype(int)  # simple threshold sum function
X_train, y_train = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.int64)

dataset = TensorDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

'''这部分代码生成了一个合成的数据集，其中有 1000 个样本和 10 个特征。目标变量 y 是基于特征和的简单阈值函数生成的。
使用 TensorDataset 和 DataLoader 将数据封装成 PyTorch 可以使用的格式，并设置批次大小和数据洗牌选项。'''

# 创建模型部分
class NormalizationModel(nn.Module):
    def __init__(self, norm_type="batch"):
        super(NormalizationModel, self).__init__()
        self.fc1 = nn.Linear(10, 50)

        if norm_type == "batch":
            self.norm = nn.BatchNorm1d(50)
        elif norm_type == "layer":
            self.norm = nn.LayerNorm(50)
        elif norm_type == "group":
            self.norm = nn.GroupNorm(5, 50)  # 5 groups

        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x

'''这个模型定义了一个简单的前馈神经网络，包含一个全连接层 fc1，一个规范化层 norm 和一个输出层 fc2。
根据构造函数传递的 norm_type 参数，可以选择不同的规范化技术。'''

# 训练函数
def train_model(norm_type):
    model = NormalizationModel(norm_type=norm_type)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 50
    losses = []

    for epoch in range(num_epochs):
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    return losses

'''这个函数负责训练模型。它接受一个规范化类型作为参数，并进行训练过程，记录并返回每一次迭代的损失。'''

# 训练和绘图
norm_types = ["batch", "layer", "group"]
results = {}

for norm_type in norm_types:
    losses = train_model(norm_type)
    results[norm_type] = losses
    plt.plot(losses, label=f"{norm_type} norm")

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Normalization Techniques Comparison")
plt.legend()
plt.show()

'''这部分代码循环遍历三种规范化技术，对每种技术分别训练模型，并绘制它们的损失曲线进行比较。
代码整体结构清晰，目的是比较不同规范化技术在训练神经网络时的效果差异。每个部分都很适合用来演示 PyTorch 框架的使用方法和神经网络中常用的规范化技术。'''


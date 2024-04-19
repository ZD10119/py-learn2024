# 计算单个张量的 L1 正则化
# 这段代码将创建一个张量 x 并计算其 L1 范数（即绝对值之和）。

import torch

# 创建一个张量
x = torch.tensor([1.0, -1.0, 2.0, -2.0], requires_grad=True)

# 计算 L1 范数
l1_norm = torch.norm(x, p=1)

print("L1 norm:", l1_norm)

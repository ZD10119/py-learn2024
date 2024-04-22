import pandas as pd

# 创建数据
data = {
    'Feature1': [1.2, 2.1, 0.9, 3.0, 1.5, 2.9, 1.0, 2.7, 1.3, 3.1],
    'Feature2': [0.7, 2.5, 0.5, 3.2, 0.4, 2.9, 1.1, 2.8, 0.8, 3.0],
    'Feature3': [0.5, 2.1, 0.4, 3.1, 0.6, 3.0, 1.0, 2.9, 0.7, 3.2],
    'Label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}

# 转换为 DataFrame
df = pd.DataFrame(data)

# 保存到 CSV 文件
df.to_csv('dataset.csv', index=False)

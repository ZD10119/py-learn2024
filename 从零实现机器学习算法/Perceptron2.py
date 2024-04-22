import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Perceptron function
def perceptron_train(x, y, z, eta, t):
    '''
    Input Parameters:
        x: data set of input features
        y: actual outputs
        z: activation function threshold
        eta: learning rate
        t: number of iterations
    '''

    # initializing the weights
    w = np.zeros(len(x[0]))
    n = 0

    # initializing additional parameters to compute sum-of-squared errors
    yhat_vec = np.ones(len(y))     # vector for predictions
    errors = np.ones(len(y))       # vector for errors (actual - predictions)
    J = []                         # vector for the SSE cost function

    while n < t:
        for i in range(0, len(x)):  # 遍历所有输入数据
            f = np.dot(x[i], w)  # 计算输入数据x[i]和权重w的点积
            if f >= z:  # 如果点积大于等于阈值z
                yhat = 1  # 预测结果为1
            else:  # 否则
                yhat = 0  # 预测结果为0
            yhat_vec[i] = yhat  # 将预测结果存储在yhat_vec向量中

            # updating the weights
            for j in range(len(w)):
                w[j] = w[j] + eta*(y[i]-yhat)*x[i][j]

        n += 1
        # computing the sum-of-squared errors
        for i in range(len(y)):
            errors[i] = (y[i]-yhat_vec[i])**2
        J.append(0.5*np.sum(errors))

    return w, J   # 'return w, J'是一个函数返回语句，用于结束函数的执行并将结果返回给调用者。

# 导入数据
df = pd.read_csv("dataset.csv")
plt.scatter(df.values[:,1], df.values[:,2], c = df['Label'], alpha=0.8)
# 将数据分割成训练集/测试集
df = df.values
np.random.seed(5)
np.random.shuffle(df)

train = df[0:int(0.7*len(df))]
test = df[int(0.7*len(df)):int(len(df))]

# 分离训练集和测试集的特征和输出
x_train = train[:, 0:3]
y_train = train[:, 3]
x_test = test[:, 0:3]
y_test = test[:, 3]

#  输入数据
z = 0.0
eta = 0.1
t = 50
perceptron_train(x_train, y_train, z, eta, t)

w = perceptron_train(x_train, y_train, z, eta, t)[0]
J = perceptron_train(x_train, y_train, z, eta, t)[1]

print("The weights are:",w)
print("The errors are:",J)

from sklearn.metrics import accuracy_score

# Assuming perceptron_train is correctly defined elsewhere
w = perceptron_train(x_train, y_train, z, eta, t)[0]

def perceptron_test(x, w, z):
    y_pred = []
    for i in range(len(x)):  # Corrected the range function
        f = np.dot(x[i], w)

        # activation function
        if f > z:
            yhat = 1
        else:
            yhat = 0
        y_pred.append(yhat)
    return y_pred

y_pred = perceptron_test(x_test, w, z)  # Removed unused parameters

print("The accuracy score is:",accuracy_score(y_test, y_pred))  # Calculate and print accuracy


# 和 scikit-learn 感知器进行比较
from sklearn.linear_model import Perceptron

# training the sklearn Perceptron
clf = Perceptron(random_state=None, eta0=0.1, shuffle=False, fit_intercept=False)
clf.fit(x_train, y_train) # Train the model with training data
y_predict = clf.predict(x_test) # Make predictions on the test data


# Display predicted labels and weights
print("Predicted labels:", y_predict)
print("sklearn weights:", clf.coef_)
print("my perceptron weights:",w)
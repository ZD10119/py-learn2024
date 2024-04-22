import numpy as np

# Perceptron function
def perceptron(x, y, z, eta, t):
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

#  输入数据    x0  x1  x2
x = [[1., 0., 0.],
     [1., 0., 1.],
     [1., 1., 0.],
     [1., 1., 1.]]

y = [1.,
     1.,
     1.,
     0.]

z = 0.0
eta = 0.1
t = 50

w = perceptron(x, y, z, eta, t)[0]
J = perceptron(x, y, z, eta, t)[1]

print("The weights are:",w)
print("The errors are:",J)
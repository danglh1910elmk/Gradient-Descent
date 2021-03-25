import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import matplotlib.animation as animation

def cost(x):
    return 0.5/N * np.linalg.norm(b - A@x, 2)**2

def grad(x):
    return 1/N * A.T @ (A@x - b)

def gradient_descent(x_init, learning_rate, iteration):
    x_list = [x_init]
    for i in range(iteration):
        x_new = x_list[-1] - learning_rate*grad(x_list[-1])
        if np.linalg.norm(grad(x_new))/N < 0.2:
            break
        x_list.append(x_new)
    return x_list


# Data
A = np.array([[2,9,7,9,11,16,25,23,22,29,29,35,37,40,46]]).T
b = np.array([[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]).T


# draw data
fig1 = plt.figure('Basic graph')
ax = plt.axes(xlim=(-10,60), ylim=(-1,20))
plt.plot(A,b,'ro')


# draw by sklearn
lr = linear_model.LinearRegression()
lr.fit(A, b)
# x0_gd = np.array([[1,46]]).T   # common
x0_gd = np.linspace(1,46,2).T   # common
y0_sklearn = lr.coef_[0][0]*x0_gd + lr.intercept_[0]
plt.plot(x0_gd,y0_sklearn)

# random initial line
x_init = np.array([[1.],[2.]])
y0_init = x_init[1][0] + x_init[0][0]*x0_gd 
plt.plot(x0_gd,y0_init, color="black")

# add ones to A
A = np.hstack((A, np.ones((A.shape[0],1))))
N = A.shape[0]

# constant
learning_rate = 0.0001
iteration = 90
x_list = gradient_descent(x_init,learning_rate,iteration)


# plot x_list
for i in range(len(x_list)):
    y0_x_list = x_list[i][1] + x_list[i][0]*x0_gd
    plt.plot(x0_gd, y0_x_list, color='g', alpha=0.5)


# draw animation
line , = ax.plot([], [], color='purple')
def update(i):
    y0_gd = x_list[i][1][0] + x_list[i][0][0]*x0_gd
    line.set_data(x0_gd, y0_gd)
    return line,

iters = np.arange(1,len(x_list),1)
line_animation = animation.FuncAnimation(fig1,update,iters,interval=20,blit=True)

# legend for graph
plt.legend(('Value in each GD iteration','Solution by formula','Initial value for GD'), loc=(0.52,0.01))
# tile
plt.title('Gradient Descent Animation')
plt.show()
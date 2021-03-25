import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Functions
def cost(x):
    return 0.5/N * np.linalg.norm((b - A@x), 2)**2

def grad(x):
    return 1/N * A.T @ (A@x - b)

def gradient_descent(x_init,learning_rate,iteration):
    x_list = [x_init]
    for i in range(iteration):
        x_new = x_list[-1] - learning_rate*grad(x_list[-1])
        if np.linalg.norm(grad(x_new),2)/N < 0.5:
            break
        x_list.append(x_new)
    return x_list


# random data
A = np.array([[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]]).T
b = np.array([[2,5,7,9,11,16,19,23,22,29,29,35,37,40,46,42,39,31,30,28,20,15,10,6]]).T


# draw data
fig1 = plt.figure('Animation')
ax = plt.axes(xlim=(-10,50), ylim=(-10,50))
plt.plot(A,b,'ro')

# Add A^2 and ones to A
N = A.shape[0]
A = np.hstack((A**2, A, np.ones((N, 1))))

# use formula
x = np.linalg.inv(A.T@A)@A.T@b

# create 1000 points in x-axis
x0_gd = np.linspace(-50,60,1000).T
y0_formula = x[0]*x0_gd**2 + x[1]*x0_gd + x[2]
plt.plot(x0_gd, y0_formula)


# random initial parabol
# x0_init = np.array([[-.1],[5.],[-7.]])  # a,b,c
x0_init = np.array([[ -2.1],[ 5.1],[-2.1]])
y0_init = x0_init[0][0]*x0_gd**2 + x0_init[1][0]*x0_gd + x0_init[2][0]
plt.plot(x0_gd,y0_init,color='black')

learning_rate = 0.000001
iteration = 1000

# GD
x_list = gradient_descent(x0_init,learning_rate,iteration)

print(len(x_list))

# plot x_list
for i in range(len(x_list)):
    y0_x_list = x_list[i][0]*x0_gd**2 + x_list[i][1]*x0_gd + x_list[i][2]
    plt.plot(x0_gd,y0_x_list,color='blue',alpha=0.5)


# draw animation
line , = ax.plot([], [], color='yellow')
def update(i):
    y0_gd = x_list[i][0]*x0_gd**2 + x_list[i][1]*x0_gd + x_list[i][2]
    line.set_data(x0_gd, y0_gd)
    return line,

iters = np.arange(1,len(x_list),1)
line_animation = animation.FuncAnimation(fig1,update,iters,interval=20,blit=True)

# legend for graph
plt.legend(('Value in each GD iteration','Solution by formula','Initial value for GD'), loc=(0.52,0.01))

# tile
plt.title('Gradient Descent Animation')
plt.show()
# %%
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
# import ipympl 
#%matplotlib widget

# %%
# data of positions by time
pos = np.array([[2, 0, 1],
       [1.08, 1.68, 2.38],
       [-0.83, 1.82, 2.49],
       [-1.97, 0.28, 2.15],
       [-1.31, -1.51, 2.59],
       [0.57, -1.91, 4.32]])

# %%
# 2 a

# grad desc. per location( or time)
def grad_desc1(x,y,vc,d0c,lrate):
    vd=np.array([0.0, 0.0, 0.0])
    d0d=np.array([0.0, 0.0, 0.0])
    n=len(x)
    for i,j in zip(x,y):
        d0d+=-2*(j-(vc*i+d0c))
        vd+= -2*i*((-vc*i-d0c)+j)
    vc=vc-lrate*vd*(1/n)
    d0c=d0c-lrate*d0d*(1/n)
    return vc, d0c

X = np.array([0,1,2,3,4,5])
vc=np.array([1,1,1])
d0c=np.array([2,0,1])
lrate=0.005
n1=len(X)
iter=10000
loss_vect=np.array([])
for i in range(iter):
    vc,d0c=grad_desc1(X,pos,vc,d0c,lrate)
    matrix = np.tile(d0c,(6, 1))
    mymat = X.T*vc.reshape((3, 1))
    yp= mymat.T + matrix
    loss=np.sum((pos-yp)**2)/n1
    loss_vect= np.append(loss_vect, loss)
print("iteration: ", i, "loss: ", loss, "v: ", vc, " d0: ",d0c)

# %%
# the loss by time
fig = plt.figure(figsize=(6, 6))
t = range(iter)
plt.plot(t,loss_vect,label='loss')
plt.title('loss by time')
plt.xlabel('time')
plt.ylabel('loss')
plt.show()

# %%
# plotting how the curves resulting from the gradient descent fit the data
t = np.linspace(0,5,100)
xxxfitfit = vc[0]*t +d0c[0]
yyyfitfit = vc[1]*t +d0c[1]
zzzfitfit = vc[2]*t +d0c[2]
x = pos[:,0]
y = pos[:,1]
z = pos[:,2]


fig, axes = plt.subplots(1, 3, figsize=(18, 6))

err_sum=0
for k in range(np.shape(pos)[0]):
    ex=pos[k][0]-(vc[0]*X[k]+d0c[0])
    ey=pos[k][1]-(vc[1]*X[k]+d0c[1])
    ez=pos[k][2]-(vc[2]*X[k]+d0c[2])
    err=ex**2+ey**2+ez**2
    err_sum=err_sum+err
print("2a) the error is", err_sum)

axes[0].plot(t, xxxfitfit, label='fit')
axes[0].scatter(X, x,label='data')
axes[0].set_title('X')
axes[0].set_xlabel('time')
axes[0].set_ylabel('x')
axes[0].legend(loc='best')

axes[1].plot(t, yyyfitfit,label='fit')
axes[1].scatter(X, y,label='data')
axes[1].set_title('Y')
axes[1].set_xlabel('time')
axes[1].set_ylabel('y')
axes[1].legend(loc='best')

axes[2].plot(t, zzzfitfit,label='fit')
axes[2].scatter(X, z,label='data')
axes[2].set_title('Z')
axes[2].set_xlabel('time')
axes[2].set_ylabel('z')
axes[2].legend(loc='best')
plt.show()

# %%
fig2 = plt.figure(figsize=(6, 6))
ax = plt.axes(projection='3d')
# # Plot the path connecting the points

ax = fig2.add_subplot(111, projection='3d')
ax.plot(x, y, z,label='trajectory')
t = np.linspace(0,5,100)

ax.plot(xxxfitfit, yyyfitfit, zzzfitfit,label='fit')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend(loc='best')
ax.set_title('trajectory and fit')
plt.show()

# %%
# 2 b


# grad desc. per location( or time)
def grad_desc1(x,y,ac,vc,d0c,lrate):
    ad=np.array([0.0, 0.0, 0.0])
    vd=np.array([0.0, 0.0, 0.0])
    d0d=np.array([0.0, 0.0, 0.0])
    n=len(x)
    for i,j in zip(x,y):
        ad+=-2*i**2*(j-(ac*i**2+vc*i+d0c))
        vd+=-2*i*((-ac*i**2-vc*i-d0c)+j)
        d0d+=-2*(j-(ac*i**2+vc*i+d0c))
    vc=vc-lrate*vd*(1/n)
    d0c=d0c-lrate*d0d*(1/n)
    ac=ac-lrate*ad*(1/n)
    return ac, vc, d0c

X = np.array([0,1,2,3,4,5])
ac=np.array([0,0,0])
vc=np.array([1,1,1])
d0c=np.array([2,0,1])
lrate=0.005
n1=len(X)
iter=10000
loss_vect=np.array([])
for i in range(iter):
    ac,vc,d0c=grad_desc1(X,pos,ac,vc,d0c,lrate)
    matrix = np.tile(d0c,(6, 1))
    mymat = X.T**2*ac.reshape((3, 1)) + X.T*vc.reshape((3, 1))
    yp= mymat.T + matrix
    loss=np.sum((pos-yp)**2)/n1
    loss_vect= np.append(loss_vect, loss)
    #print("iteration: ", i, "loss: ", loss, "a:", ac, "v: ", vc, " d0: ",d0c)
print("iteration: ", i, "loss: ", loss, "a:", ac, "v: ", vc, " d0: ",d0c)

# %%
# the loss by time
fig = plt.figure(figsize=(6, 6))
t = range(iter)
plt.plot(t,loss_vect,label='loss')
plt.title('loss by time')
plt.legend(loc='best')
plt.xlabel('time')
plt.ylabel('loss')
plt.show()


# %%
# plotting how the curves resulting from the gradient descent fit the data
t = np.linspace(0,5,100)
xfitfit = ac[0]*t**2 + vc[0]*t +d0c[0]
yfitfit = ac[1]*t**2 + vc[1]*t +d0c[1]
zfitfit = ac[2]*t**2 + vc[2]*t +d0c[2]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

err_sum=0
for k in range(np.shape(pos)[0]):
    ex=pos[k][0]-(ac[0]*X[k]**2+vc[0]*X[k]+d0c[0])
    ey=pos[k][1]-(ac[1]*X[k]**2+vc[1]*X[k]+d0c[1])
    ez=pos[k][2]-(ac[2]*X[k]**2+vc[2]*X[k]+d0c[2])
    err=ex**2+ey**2+ez**2
    err_sum=err_sum+err
print("2b) the error is", err_sum)

axes[0].plot(t, xfitfit,label='fit')
axes[0].scatter(X, pos[:,0],label='data')
axes[0].set_title('X')
axes[0].set_xlabel('time')
axes[0].set_ylabel('x')
axes[0].legend(loc='best')


axes[1].plot(t, yfitfit,label='fit')
axes[1].scatter(X, pos[:,1],label='data')
axes[1].set_title('Y')
axes[1].set_xlabel('time')
axes[1].set_ylabel('y')
axes[1].legend(loc='best')

axes[2].plot(t, zfitfit,label='fit')
axes[2].scatter(X, pos[:,2],label='data')
axes[2].set_title('Z')
axes[2].set_xlabel('time')
axes[2].set_ylabel('z')
axes[2].legend(loc='best')
plt.show()

# %%
# 3d plot
fig2 = plt.figure(figsize=(6, 6))
ax = plt.axes(projection='3d')
# # Plot the path connecting the points

ax = fig2.add_subplot(111, projection='3d')
ax.plot(x, y, z, label='data')
t = np.linspace(0,5,100)

ax.plot(xfitfit, yfitfit, zfitfit, label='parametric curve')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend(loc='best')
ax.set_title('trajectory and fit')
plt.show()


# %%
#2 c

# plotting how the curves resulting from the gradient descent fit the data
t = np.linspace(0,6,100)
xxfitfit = ac[0]*t**2 + vc[0]*t +d0c[0]
yyfitfit = ac[1]*t**2 + vc[1]*t +d0c[1]
zzfitfit = ac[2]*t**2 + vc[2]*t +d0c[2]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
X = [0,1,2,3,4,5,6]

axes[0].plot(t, xxfitfit, label='x-fit')
axes[0].scatter(X, np.append(pos[:,0],xxfitfit[len(xxfitfit)-1]), label='x-data')
axes[0].scatter(6, xxfitfit[len(xxfitfit)-1],label='x-prediction')
axes[0].set_title('X fit')
axes[0].set_xlabel('time')
axes[0].set_ylabel('x')
axes[0].legend(loc='best')

axes[1].plot(t, yyfitfit,label='y-fit')
axes[1].scatter(X, np.append(pos[:,1],yyfitfit[len(yyfitfit)-1]), label='y-data')
axes[1].scatter(6, yyfitfit[len(yyfitfit)-1], label='y-prediction')
axes[1].set_title('Y fit')
axes[1].set_xlabel('time')
axes[1].set_ylabel('y')
axes[1].legend(loc='best')

axes[2].plot(t, zzfitfit,label='z-fit')
axes[2].scatter(X, np.append(pos[:,2],zzfitfit[len(zzfitfit)-1]), label='z-data')
axes[2].scatter(6, zzfitfit[len(zzfitfit)-1], label='z-prediction')
axes[2].set_title('Z fit')
axes[2].set_xlabel('time')
axes[2].set_ylabel('z')
axes[2].legend(loc='best')



plt.show()

# %%
fig2 = plt.figure(figsize=(6, 6))
ax = plt.axes(projection='3d')
# # Plot the path connecting the points

ax = fig2.add_subplot(111, projection='3d')
ax.plot(x, y, z,label='data')
tt = np.linspace(0,6,100)

ax.plot(xxfitfit, yyfitfit, zzfitfit, label='fit')
ax.scatter(xxfitfit[len(xfitfit)-1], yyfitfit[len(yyfitfit)-1], zzfitfit[len(zzfitfit)-1],label='prediction')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend(loc='best')
ax.set_title('trajectory and fit')
plt.show()
print("the coordinates of the new point are: ",(xxfitfit[len(xfitfit)-1], yyfitfit[len(yyfitfit)-1], zzfitfit[len(zzfitfit)-1]))


# %%
D = np.ones((6, 4))
for i in range(len(D)):
    D[i][1] *= i
    D[i][2] *= i**2
    D[i][3] *= i**3

a0, a1, a2, a3 = np.linalg.inv(D.T@D)@D.T@x
b0, b1, b2, b3 = np.linalg.inv(D.T@D)@D.T@y
c0, c1, c2, c3 = np.linalg.inv(D.T@D)@D.T@z

t = np.linspace(0,5,100)
def f(a0, a1, a2, a3, t):
     return a0 + a1*t + a2*t**2 +  a3*t**3
fig = plt.figure(figsize=(5, 5))
ax = plt.axes(projection='3d')
# # Plot the path connecting the points

xfit = f(a0, a1, a2, a3, t)
yfit = f(b0, b1, b2, b3, t)
zfit = f(c0, c1, c2, c3, t)

ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z)
ax.plot(xfit, yfit, zfit)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# %%
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

# Generate some 3D data

# Create a 3D figure and axis

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the data

# ax = fig2.add_subplot(111, projection='3d')
ax.plot(x, y, z,label='datapoints sequentially connected by lines')

ax.plot(xxxfitfit, yyyfitfit, zzzfitfit, label='fit with constant veclocity')
ax.plot(xfit, yfit, zfit, label='trajectory from polynomial regression to power 3')
t = np.linspace(0,6,100)
ax.plot(xxfitfit, yyfitfit, zzfitfit, label='fit with constant acceleration and with extra second')
ax.scatter(xxfitfit[len(xfitfit)-1], yyfitfit[len(yyfitfit)-1], zzfitfit[len(zzfitfit)-1],label='prediction')

# ax.plot(xxfitfit, yyfitfit, zzfitfit, label='fit')
# ax.plot(x, y, z, label='data')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend(loc='best')
ax.set_title('3D plot')

# Define the update function for the animation:
def update(num):
    ax.view_init(elev=10., azim=num)

# Create the animation:
ani = animation.FuncAnimation(fig, update, frames=360, interval=50)
plt.show()

# Save the animation as a GIF file:
# ani.save('3D_plot_rotation.gif', writer='imagemagick', fps=60)
# convert to mp4:
# !ffmpeg -i 3D_plot_rotation.gif -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" my_video.mp4



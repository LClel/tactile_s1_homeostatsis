import numpy as np
import matplotlib.pyplot as plt
import somatotopy_python as sp

### Single unit homeostasis test

num = 50000

# set up homeostasis parameters
av_act = np.zeros(num*3)
theta = np.zeros(num*3)
act = np.zeros(num*3)
beta = .991
lmbda = .001
mu = .024
initTheta = .25

# input set 1
m = .024 # mean of input
s = 1 # SD of input

in1 = np.random.normal(m,s,num)
in1_mu = [m] * num


# input set 2
m = .05 # mean of input
s = 1 # SD of input

in2 = np.random.normal(m,s,num)
in2_mu = [m] * num


# input set 3
m = .001 # mean of input
s = 1 # SD of input

in3 = np.random.normal(m,s,num)
in3_mu = [m] * num


# concat input sets
inputs = np.concatenate([in1,in2,in3])
mus = np.concatenate([in1_mu,in2_mu,in3_mu])


act,theta,av_act = sp.singleunithomeostasis(inputs=inputs,num=50000,av_act=av_act,theta=theta,initTheta=.25,
                                         act=act)

plt.figure(figsize=(20,20))

# plot of threshold changes over time
plt.subplot(5,1,1)
plt.plot(theta)
plt.ylabel('Theta')
plt.xlabel('Iteration')
plt.title('Threshold over time')

# plot of average activation changes over time, with means of inputs plotted
plt.subplot(5,1,2)
plt.plot(av_act)
plt.plot(mus)
plt.ylabel('Average activation')
plt.xlabel('Iteration')
plt.title('Average activation over time')

plt.subplot(5,1,3)
act[act<0] = 0
plt.plot(act)
plt.axvline(x=num,linestyle='--',color='black')
plt.axvline(x=(num*2),linestyle='--',color='black')
plt.ylabel('Activation')
plt.xlabel('Iteration')
plt.title('Activation of the unit over time')

plt.subplot(5,1,4)
plt.plot(theta)
plt.plot(av_act)
plt.plot(mus)
plt.axhline(y=mu,color='r',label='Target activation')
plt.axvline(x=num,linestyle='--',color='black')
plt.axvline(x=(num*2),linestyle='--',color='black')

plt.show()
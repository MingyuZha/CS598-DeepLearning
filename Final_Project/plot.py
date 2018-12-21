import numpy as np
import matplotlib.pyplot as plt
a = np.load('MSCOCO_resnetloss.npy')
b = np.load('MSCOCO_Resnet512loss.npy')
c = np.load('MSCOCO_resnet2000loss.npy')
a = list(a)
b = list(b)
c = list(c)
idxa = [i for i in range(len(a))]
idxb = [i for i in range(len(b))]
idxc = [i for i in range(len(c))]
plt.figure(figsize=(20,10))

plt.subplot(1,2,1)
plt.title("Loss on training set")
plt.plot(idxa, a, label='resnet1000')
plt.plot(idxb, b, label='resnet512')
plt.plot(idxc, c, label='resnet2000')
plt.legend()


ax = np.load('MSCOCO_resneteval_loss.npy')
bx = np.load('MSCOCO_Resnet512eval_loss.npy')
cx = np.load('MSCOCO_resnet2000eval_loss.npy')

ax = list(ax)
bx = list(bx)
cx = list(cx)

idxax = [i for i in range(len(ax))]
idxbx = [i for i in range(len(bx))]
idxcx = [i for i in range(len(cx))]


plt.subplot(1,2,2)
plt.title("Loss on validation set")
plt.plot(idxax, ax, label='resnet1000')
plt.plot(idxbx, bx, label='resnet512')
plt.plot(idxcx, cx, label='resnet2000')

plt.legend()
plt.savefig('figure1.png')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
selected = ['val_901', 'val_5714', 'val_9759', 'val_8142', 'val_6885']
selected_labels = ['n02423022', 'n01950731', 'n07753592', 'n03970156', 'n04146614']
plt.subplots_adjust(left=0.5, bottom=1, right=0.5, top=0.8, wspace=1, hspace=1)
for i in range(10):
    plt.subplot(3,4,i+1)
    if (i == 0):
        im = mpimg.imread('./tiny-imagenet-200/val/images/'+selected[0]+'.JPEG')
        plt.imshow(im)
        plt.xlabel(selected_labels[0])
        plt.title("Query Image")
    im = mpimg.imread('./saved_images/'+selected[0]+'_'+str(i)+'.JPEG')
    plt.imshow(im)
# plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
plt.show()
import spectral as spy
import matplotlib.pyplot as plt
from scipy.io import loadmat

# input_image = loadmat('D:\\rmdmy\\datasets\\Salinas\\salinas.mat')['HSI_original']
# gt = loadmat('D:\\rmdmy\\datasets\\Salinas\\salinas_gt.mat')['Data_gt']
input_image = loadmat('D:\\rmdmy\\datasets\\Pavia\\paviaU.mat')['paviaU']
gt = loadmat('D:\\rmdmy\\datasets\\Pavia\\paviaU_gt.mat')['Data_gt']
# print(gt.keys())
# spy.save_rgb('rgb.jpg', input_image, [100,98,13])
spy.save_rgb('rgb.jpg', input_image, [57,34,3])
# view1 = spy.imshow(data=input_image, bands=[100,98,13], title="img")  # 图像显示
#
# view2 = spy.imshow(classes=gt, title="gt")  # 地物类别显示
#
# view3 = spy.imshow(data=input_image, bands=[69, 27, 11], classes=gt)
# view3.set_display_mode("overlay")
# view3.class_alpha = 0.3  # 设置类别透明度为0.3
# #
# spy.view_cube(input_image, bands=[69, 27, 11])  # 显示后会打印相应功能及操作
# #
# pc = spy.principal_components(input_image)  # N维特征显示 view_nd与view_cube需要ipython 命令行输入：ipython --pylab
# xdata = pc.transform(input_image)  # 把数据转换到主成分空间
# spy.view_nd(xdata[:, :, :15], classes=gt)

# plt.pause(60)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class ToolImage:

    @staticmethod
    def _show_images(images):  # 定义画图工具
        images = np.reshape(images, [images.shape[0], -1])
        sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
        sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

        fig = plt.figure(figsize=(sqrtn, sqrtn))
        gs = gridspec.GridSpec(sqrtn, sqrtn)
        gs.update(wspace=0.05, hspace=0.05)
        for i, img in enumerate(images):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(img.reshape([sqrtimg, sqrtimg]))

        return

    @staticmethod
    def _deprocess_img(x):
        return (x + 1.0) / 2.0

    @staticmethod
    def display(images):
        images = ToolImage._deprocess_img(images)
        ToolImage._show_images(images[:16])
        plt.show()


display = ToolImage.display

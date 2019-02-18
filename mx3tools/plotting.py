import numpy as np
import ipywidgets
import PIL
import matplotlib.animation as mpla
import matplotlib.pyplot as plt
import matplotlib.transforms as mplt
import IPython


def image_slider(data, plane='xy', component=2, scale=1, rotate=0, kind='pil'):

    if plane == 'xy':
        i_max = data.shape[0] - 1
    elif plane == 'xz':
        i_max = data.shape[1] - 1
    elif plane == 'yz':
        i_max = data.shape[2] - 1
    else:
        raise ValueError(f'Invalid viewpoint: {plane}')

    data = np.uint8((data + 1)*0.5*255)

    @ipywidgets.interact(i=(0, i_max))
    def _show(i):
        if plane == 'xy':
            im = PIL.Image.fromarray(data[i, :, :, component])
        elif plane == 'xz':
            im = PIL.Image.fromarray(data[:, i, :, component])
        elif plane == 'yz':
            im = PIL.Image.fromarray(data[:, :, i, component])

        return im.rotate(rotate).resize((np.array(im.size)*scale).astype(int))

    return _show


def frame_slider_2D(data, plane='xy', component=2, scale=1, rotate=0, kind='pil'):

    data = np.uint8((data + 1)*0.5*255)

    if kind.lower() in ['pil', 'pillow']:
        @ipywidgets.interact(i=(0, data.shape[0] - 1))
        def _show(i):
            if plane == 'xy':
                im = PIL.Image.fromarray(data[i, 0, :, :, component])
            elif plane == 'xz':
                raise NotImplementedError
            elif plane == 'yz':
                raise NotImplementedError

            return im.rotate(rotate).resize((np.array(im.size)*scale).astype(int))

    if kind.lower() in ['mpl', 'matplotlib']:

        raise NotImplementedError

        plt.ioff()
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # @ipywidgets.interact(i=(0, data.shape[0] - 1))

        slider = ipywidgets.IntSlider(value=0, min=0, max=data.shape[0] - 1)

        def _show(i):
            # plt.clf()
            if plane == 'xy':
                im = ax.imshow(data[i, 0, :, :, component], cmap='RdBu_r', origin='lower')
            elif plane == 'xz':
                raise NotImplementedError
            elif plane == 'yz':
                raise NotImplementedError

            trans_data = ax.transData + mplt.Affine2D().scale(scale).rotate_deg(rotate)
            im.set_transform(trans_data)
            fig.canvas.draw()
            fig.canvas.flush_events()
            return

        slider.observe(_show, names='value')
        ipywidgets.VBox([fig.canvas, slider])

    return _show


def text_slider(data, plane='xy', component=2):

    if plane == 'xy':
        i_max = data.shape[0] - 1
    elif plane == 'xz':
        i_max = data.shape[1] - 1
    elif plane == 'yz':
        i_max = data.shape[2] - 1
    else:
        raise ValueError(f'Invalid viewpoint: {plane}')

    s = ipywidgets.IntSlider(value=0, min=0, max=i_max)

    def _show(i):
        if plane == 'xy':
            pprint_array(data[i, :, :, component])
            return
        elif plane == 'xz':
            pprint_array(data[:, i, :, component])
            return
        elif plane == 'yz':
            pprint_array(data[:, :, i, component])
            return

    out = ipywidgets.interactive_output(_show, {'i': s})
    ipywidgets.VBox([s, out])

    return


def pprint_array(arr):
    for i in range(arr.shape[0]):
        print(' '.join(list(arr[i].astype(str))))
    return


def mpl_animation(data, plane='xy', component=2):

    fig, ax = plt.subplots()

    def init():
        im = ax.imshow(data[0, :, :, 0, component], origin='lower', cmap='RdBu_r')
        return im

    def animate(i):
        ax.clear()
        im = ax.imshow(data[i, :, :, 0, component], origin='lower', cmap='RdBu_r')
        return im

    anim = mpla.FuncAnimation(fig, animate, init_func=init, frames=data.shape[0], interval=20, blit=True)
    IPython.display.HTML(anim.to_jshtml())

    return

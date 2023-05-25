import torch as th
from matplotlib import animation, colors, pyplot as plt
import numpy as np
import cv2

import os
import sys
import time
from typing import Tuple
import multiprocessing as mp

def show_flow_net(*tensors, grid_x=None, grid_y=None):

    fig  = plt.figure(constrained_layout=True, figsize=(20, 8))
    axes = fig.subplots(1, len(tensors))

    def animate(t):
        for i in range(len(tensors)):
            tensor = tensors[i]
            if i == 1:
                length = tensor[0,t].transpose(1,2).detach().cpu().numpy() # TODO need to flip horizontaly
                tensor = th.sqrt(th.sum(tensor**2, dim=2)).unsqueeze(dim=2)
            tensor = tensor.detach().cpu().numpy()
            tensor = tensor[0, t]

            axis = axes[i] if len(tensors) > 1 else axes
            axis.clear()
            axis.imshow(tensor.transpose(1,2,0))
            if i == 1:
                print(grid_x.shape, length[0,::4,::4].shape)
                axis.quiver(grid_x, grid_y, length[0,::4,::4]*-1, length[1,::4,::4], scale=60, width=0.0010)

        axes[0].title.set_text('input')
        axes[1].title.set_text('flow')
        axes[2].title.set_text('prediction')
        axes[3].title.set_text('error')
        plt.savefig(f'flow-net-{t}.png')

    anim = animation.FuncAnimation(fig, animate, frames=tensors[0].shape[1], repeat=False)
    plt.show()
    plt.close()

def show_tensors(*tensors, file=None):

    fig  = plt.figure(constrained_layout=True, figsize=(20, 8))
    size = int(np.ceil(np.sqrt(len(tensors))))
    axes = fig.subplots(size, size) 
    #axes = fig.subplots(1, size) 


    def animate(t):
        for i in range(len(tensors)):
            tensor = tensors[i]
            tensor = tensor.detach().cpu().numpy()
            tensor = tensor[0, t]

            axis = axes[i//size, i%size] if len(tensors) > 1 else axes
            #axis = axes[i] if len(tensors) > 1 else axes
            axis.clear()
            if tensor.shape[0] == 1:
                #axis.imshow(tensor.transpose(1,2,0), cmap='gray', vmin=0, vmax=1)
                #axis.imshow(tensor.transpose(1,2,0), vmin=0, vmax=1)
                axis.imshow(tensor.transpose(1,2,0))
            else:
                #r,g,b = tensor[0], tensor[1], tensor[2]
                #tensor[0], tensor[1], tensor[2] = g,r,b
                #tensor[0], tensor[1], tensor[2] = g,b,r
                #tensor[0], tensor[1], tensor[2] = b,g,r
                #tensor[0], tensor[1], tensor[2] = b,r,g
                #tensor[0], tensor[1], tensor[2] = r,b,g
                #tensor[0], tensor[1], tensor[2] = r,g,b
                #cv2.imwrite(f'img{t}.{i}.jpg', tensor.transpose(1,2,0) * 255)

                tensor = tensor.transpose(1,2,0)
                tensor = tensor[...,::-1]
                axis.imshow(tensor)
        
        if file != None:
            plt.savefig(f'{file}-{t:03d}.png')

    anim = animation.FuncAnimation(fig, animate, frames=tensors[0].shape[1])
    plt.show()
    plt.close()



def show_bias(*biases, title='Local Bias'):
    """
    Shows a local bias
    :param biases: List of local biases
    """
    fig, axes = plt.subplots(1, len(biases))

    for i in range(len(biases)):
        bias = biases[i]
        bias = bias.detach().cpu().numpy()
        bias = bias[0]
        bias = bias.transpose(1,2,0)
        bias = bias[...,::-1]

        axis = axes[i] if len(biases) > 1 else axes
        axis.imshow(bias)

    plt.suptitle(title)

    plt.show()
    plt.close()

def animate_lstm(states, title="c states"):
    """
    Shows a local bias
    :param biases: List of local biases
    """

    rows = int(np.sqrt(states.size(2)))
    fig, axes = plt.subplots(rows, states.size(2) // rows) 
    plt.suptitle(title)

    def animate(t):
        for i in range(rows * (states.size(2) // rows)):
            s = states[t,0,i]
            s = s.detach_().cpu().numpy()

            axes[i // rows, i % rows].imshow(s)

    anim = animation.FuncAnimation(fig, animate, frames=states.size(0))
    plt.show()
    plt.close()

def cov(X):
    D = X.shape[-1]
    mean = th.mean(X, dim=-1).unsqueeze(-1)
    X = X - mean
    return 1/(D-1) * X.unsqueeze(1) @ X.unsqueeze(0)

def show_cov(states, title="covariance c states", file=None):
    """
    Shows a local bias
    :param biases: List of local biases
    """
    rows = int(np.sqrt(states.size(2)))

    fig = plt.figure(constrained_layout=True, figsize=(10, 4))
    subfigs = fig.subfigures(1, 2, wspace=0.07) 

    cov_axes = subfigs[0].subplots(1,1, sharey=True)
    axes = subfigs[1].subplots(rows, states.size(2) // rows, sharey=True)
    plt.suptitle(title)

    def animate(t):
        s = th.mean(states[t,0], dim=(1,2))
        s = cov(s)
        s = s.detach_().cpu().numpy()

        cov_axes.imshow(s)

        for i in range(rows * (states.size(2) // rows)):
            s = states[t,0,i]
            s = s.detach_().cpu().numpy()

            axes[i // rows, i % rows].imshow(s)

        if file != None:
            plt.savefig(f'{file}-{t}.png')


    anim = animation.FuncAnimation(fig, animate, frames=states.size(0), repeat=(file == None))
    plt.show()
    plt.close()

def show_hidden(input, output, states, batch = 0, title="hidden", file=None):
    rows = len(states)
    cols = len(states[0])

    fig = plt.figure(constrained_layout=True, figsize=(10, 6))
    subfigs = fig.subfigures(1, 3, wspace=0.07) 

    input_axes  = subfigs[0].subplots(1,1, sharey=True)
    output_axes = subfigs[2].subplots(1,1, sharey=True)
    axes = subfigs[1].subplots(rows, cols, sharey=True)
    plt.suptitle(title)

    def animate(t):
        input_axes.imshow(input[batch,t,0].detach().cpu().numpy(), cmap='Blues_r')
        output_axes.imshow(output[batch,t,0].detach().cpu().numpy(), cmap='Blues_r')

        for r in range(rows):
            for c in range(cols):
                s = states[r][c][batch,t,0]
                s = s.detach().cpu().numpy()

                if cols == 1:
                    axes[r].imshow(s, cmap='Blues_r')
                else:
                    axes[r, c].imshow(s, cmap='Blues_r')

        if file != None:
            plt.savefig(f'{file}-{t}.png')


    anim = animation.FuncAnimation(fig, animate, frames=input.size(1), repeat=(file == None))
    plt.show()
    plt.close()

def save_gpnet(input, output, states, batch = 0, title="hidden", file=None):

    plt.ioff()
    pid = os.fork()

    if pid > 0:
        os.waitpid(pid, 0)
        return

    rows = len(states)
    cols = len(states[0])

    fig = plt.figure(constrained_layout=True, figsize=(20, 8))
    subfigs = fig.subfigures(1, 3, wspace=0.07, width_ratios=(4,8,4))

    input_axes  = subfigs[0].subplots(len(input), 1, sharey=True)
    output_axes = subfigs[2].subplots(len(output),1, sharey=True)
    axes = subfigs[1].subplots(rows, cols, sharey=True)
    plt.suptitle(title)

    axes[0,0].title.set_text('input mask')
    axes[1,0].title.set_text('masked position input')
    axes[2,0].title.set_text('masked gestalt input')
    axes[3,0].title.set_text('position')
    axes[4,0].title.set_text('mask')
    axes[5,0].title.set_text('object')

    t = 0
    if input[0].shape[2] == 1:
        input_axes.imshow(input[0][batch,t,0], cmap='Blues_r', vmin=0, vmax=1)
        input_axes.get_xaxis().set_visible(False)
        input_axes.get_yaxis().set_visible(False)
    else:
        if len(input) == 3:
            input_axes[0].imshow(input[0][batch,t,::-1].transpose(1,2,0))
            input_axes[1].imshow(input[1][batch,t,::-1].transpose(1,2,0))
            input_axes[2].imshow(input[2][batch,t,::-1].transpose(1,2,0))
            input_axes[0].get_xaxis().set_ticks([])
            input_axes[1].get_xaxis().set_ticks([])
            input_axes[2].get_xaxis().set_ticks([])
            input_axes[0].get_yaxis().set_ticks([])
            input_axes[0].title.set_text('unprocessed input')
            input_axes[1].title.set_text('RMSE(input, background)')
            input_axes[2].title.set_text('indentifed objects (encoder)')
        else:
            input_axes.imshow(input[0][batch,t].transpose(1,2,0))

    if output[0].shape[2] == 1:
        output_axes.imshow(output[0][batch,t,0], cmap='Blues_r', vmin=0, vmax=1)
    else:
        if len(output) == 3:
            output_axes[0].imshow(output[0][batch,t,::-1].transpose(1,2,0))
            output_axes[1].imshow(output[1][batch,t,::-1].transpose(1,2,0))
            output_axes[2].imshow(output[2][batch,t,::-1].transpose(1,2,0))
            output_axes[0].get_xaxis().set_ticks([])
            output_axes[1].get_xaxis().set_ticks([])
            output_axes[2].get_xaxis().set_ticks([])
            output_axes[0].get_yaxis().set_ticks([])
            output_axes[0].title.set_text('predicted next frame')
            output_axes[1].title.set_text('predicted next frame (background subtracted)')
            output_axes[2].title.set_text('error')
        else:
            output_axes.imshow(output[0][batch,t].transpose(1,2,0))

    for r in range(rows):
        for c in range(cols):
            s = states[r][c][batch,-1]
            if states[r][c].shape[1] > t:
                s = states[r][c][batch,t]

            if s.shape[0] == 1:
                s = s[0]
                if cols == 1:
                    axes[r].imshow(s, cmap='Blues_r', vmin=0, vmax=1)
                    axes[r].get_xaxis().set_ticks([])
                    axes[r].get_yaxis().set_ticks([])
                else:
                    axes[r, c].imshow(s, cmap='Blues_r', vmin=0, vmax=1)
                    axes[r, c].get_xaxis().set_ticks([])
                    axes[r, c].get_yaxis().set_ticks([])
            else:
                if cols == 1:
                    axes[r].imshow(s.transpose(1,2,0))
                    axes[r].get_xaxis().set_ticks([])
                    axes[r].get_yaxis().set_ticks([])
                else:
                    axes[r, c].imshow(s.transpose(1,2,0))
                    axes[r, c].get_xaxis().set_ticks([])
                    axes[r, c].get_yaxis().set_ticks([])

    plt.savefig(f'{file}.png')
    os._exit(0)


def show_autoencoder_unet(input, output, states, batch = 0, title="hidden", file=None):
    rows = len(states)
    cols = len(states[0])

    fig = plt.figure(constrained_layout=True, figsize=(20, 8))
    subfigs = fig.subfigures(1, 3, wspace=0.07, width_ratios=(4,8,4)) 

    input_axes  = subfigs[0].subplots(len(input), 1, sharey=True)
    output_axes = subfigs[2].subplots(len(output),1, sharey=True)
    axes = subfigs[1].subplots(rows, cols, sharey=True)
    plt.suptitle(title)

    axes[0,0].title.set_text('input mask')
    axes[1,0].title.set_text('masked position input')
    axes[2,0].title.set_text('masked gestalt input')
    axes[3,0].title.set_text('position')
    axes[4,0].title.set_text('mask')
    axes[5,0].title.set_text('object')

    def animate(t):
        for i in range(len(input_axes)):
            input_axes[i].imshow(input[i][batch,t,::-1].transpose(1,2,0), cmap = 'Blues_r' if input[i].shape[2] == 1 else None, vmin=0, vmax=1)
            input_axes[i].get_xaxis().set_ticks([])
            input_axes[i].get_yaxis().set_ticks([])

        if len(input) == 3:
            input_axes[0].title.set_text('unprocessed input')
            input_axes[1].title.set_text('RMSE(input, background)')
            input_axes[2].title.set_text('indentifed objects (encoder)')
        else:
            input_axes[0].title.set_text('input')
            input_axes[i].title.set_text('indentifed objects (encoder)')

        for i in range(len(output_axes)):
            output_axes[i].imshow(output[i][batch,t,::-1].transpose(1,2,0), cmap = 'Blues_r' if output[i].shape[2] == 1 else None)
            output_axes[i].get_xaxis().set_ticks([])
            output_axes[i].get_yaxis().set_ticks([])

        if len(output) == 3:
            output_axes[0].title.set_text('predicted next frame')
            output_axes[1].title.set_text('predicted next frame (background subtracted)')
            output_axes[2].title.set_text('error')
        else:
            output_axes[0].title.set_text('predicted next frame')
            output_axes[1].title.set_text('error')

        for r in range(rows):
            for c in range(cols):
                s = states[r][c][batch,-1]
                if states[r][c].shape[1] > t:
                    s = states[r][c][batch,t]

                if s.shape[0] == 1:
                    s = s[0]
                    if cols == 1:
                        axes[r].imshow(s, cmap='Blues_r', vmin=0, vmax=1)
                        axes[r].get_xaxis().set_ticks([])
                        axes[r].get_yaxis().set_ticks([])
                    else:
                        axes[r, c].imshow(s, cmap='Blues_r', vmin=0, vmax=1)
                        axes[r, c].get_xaxis().set_ticks([])
                        axes[r, c].get_yaxis().set_ticks([])
                else:
                    if cols == 1:
                        axes[r].imshow(s.transpose(1,2,0))
                        axes[r].get_xaxis().set_ticks([])
                        axes[r].get_yaxis().set_ticks([])
                    else:
                        axes[r, c].imshow(s.transpose(1,2,0))
                        axes[r, c].get_xaxis().set_ticks([])
                        axes[r, c].get_yaxis().set_ticks([])

        if file != None:
            plt.savefig(f'{file}-{t}.png')


    anim = animation.FuncAnimation(fig, animate, frames=input[0].shape[1], repeat=(file == None))
    plt.show()
    plt.close()


def show_out(out: th.Tensor, label: th.Tensor, batch=0, save=False, teacher_forcing=-1):
    """
    Shows the output and label
    :param out: Output tensor
    :param label: Label tensor
    :param batch: Number of the batch used
    :param save: Saves the animation as out.mp4
    :param teacher_forcing: Number of shown teacher forcing steps for an information
    """
    _out = np.array(out.detach().cpu().numpy(), dtype=np.float32)
    _label = np.array(label.detach().cpu().numpy(), dtype=np.float32)

    out = _out[batch]
    label = _label[batch]

    fig, axes = plt.subplots(1, 3)

    def animate(_i):
        axes[0].clear()
        axes[1].clear()
        axes[2].clear()

        suptitle_teacher_forcing = " (teacher forcing)" if _i < teacher_forcing else ""
        plt.suptitle(f'Prediction for time step {_i + 1}/{len(out)}{suptitle_teacher_forcing}')

        axes[0].set_title('difference')
        axes[1].set_title('Output')
        axes[2].set_title('Output label')

        axes[0].imshow(out[_i, 0] - label[_i, 0], cmap='Blues_r', vmin=-1, vmax=1)
        axes[1].imshow(out[_i, 0], cmap='Blues_r', vmin=-1, vmax=1)
        axes[2].imshow(label[_i, 0], cmap='Blues_r', vmin=-1, vmax=1)

    anim = animation.FuncAnimation(fig, animate, frames=len(out))

    if save:
        try:
            plt.rcParams['animation.ffmpeg_path'] = os.environ["FFMpeg"]
            anim.save(f'out.mp4', animation.FFMpegWriter())
        except KeyError:
            print(f'Was not able to save the movie. Please set the "FFMpeg" environment variable to your ffmpeg path',
                  file=sys.stderr)
        except FileNotFoundError:
            print(f'Was not able to save the movie. '
                  f'The environment variable "FFMpeg" with value "{os.environ["FFMpeg"]}" is not a path to ffmpeg',
                  file=sys.stderr)
    plt.show()

    plt.close()

def show_autoencoder(input: th.Tensor, latent: th.Tensor, output: th.Tensor, batch=0, save=False):
    """
    Shows the output and label
    :param out: Output tensor
    :param label: Label tensor
    :param batch: Number of the batch used
    :param save: Saves the animation as out.mp4
    """
    input = input.detach().cpu().numpy()[batch]
    output = output.detach().cpu().numpy()[batch]

    n_latent = len(latent)
    fig, axes = plt.subplots(1, 2+n_latent)

    def animate(_i):
        for i in range(2+n_latent):
            axes[i].clear()

        axes[0].set_title('Input')
        for i in range(n_latent):
            axes[i+1].set_title('Latent state {}'.format(i))

        axes[n_latent+1].set_title('Output')

        axes[0].imshow(input[_i, 0], cmap='Blues_r', vmin=-1, vmax=1)
        for i in range(n_latent):
            axes[i+1].imshow(latent[i].detach().cpu().numpy()[batch][_i, 0], cmap='Blues_r', vmin=-1, vmax=1)
        axes[n_latent+1].imshow(output[_i, 0], cmap='Blues_r', vmin=-1, vmax=1)

    anim = animation.FuncAnimation(fig, animate, frames=len(output))

    if save:
        plt.rcParams['animation.ffmpeg_path'] = "python\\Lib\\site-packages\\imageio_ffmpeg\\binaries\\ffmpeg-win64-v4.2.2.exe"
        anim.save(f'out.mp4', animation.FFMpegWriter())
    plt.show()

    plt.close()


def show_amplitude(out: th.Tensor, label: th.Tensor, b=0, c=0):
    """
    Shows the output and label
    :param out: Output tensor
    :param label: Label tensor
    """

    out = out[b, :, c, out.shape[3] // 2, out.shape[4] // 2]
    label = label[b, :, c, label.shape[3] // 2, label.shape[4] // 2]

    _out = out.detach().cpu().numpy()
    _label = label.detach().cpu().numpy()

    timesteps = len(_label)

    fig, ax = plt.subplots(1, 1, figsize=[8, 2])
    ax.plot(range(timesteps), _label)
    ax.plot(range(timesteps), _out)
    ax.set_xlabel("Time")
    ax.set_ylabel("Wave amplitude")
    ax.set_xlim([0, timesteps - 1])
    plt.tight_layout()
    plt.show()
    plt.close()


def show_loss(output: th.Tensor, label: th.Tensor):
    loss = (output - label)**2
    loss = th.mean(loss, dim=(0, 2, 3, 4))
    loss = loss.detach().cpu().numpy()

    fig, axis = plt.subplots(1, 1)
    axis.plot(loss, label='loss')
    axis.set_yscale('log')
    axis.set_xlabel('Time steps')
    axis.set_ylabel('MSE')
    axis.set_yscale('log')

    plt.title('Mean network loss')
    plt.legend()
    plt.show()
    plt.close()


def show_quantity(output: th.Tensor, label: th.Tensor, b=0):
    output = output.detach().cpu().numpy()
    label = label.detach().cpu().numpy()

    output = output[b].reshape((output.shape[1], -1))
    label = label[b].reshape((label.shape[1], -1))

    mean_output = output.mean(axis=-1)
    mean_label = label.mean(axis=-1)

    fig, axis = plt.subplots(1, 1)
    axis.plot(mean_output, label='Output')
    axis.plot(mean_label, label='Label')
    axis.set_xlabel('Time steps')
    axis.set_ylabel('Quantity')

    plt.title('Mean network quantity')
    plt.legend()
    plt.show()

    plt.close()


def show_activity(activity: th.Tensor, label: th.Tensor):
    """
    Shows the activity controller, label
    :param activity: Activity tensor
    :param label: Label tensor
    """
    activity = activity.detach().cpu().numpy()
    label = label.detach().cpu().numpy()

    activity = activity[:, 0]
    label = label[0]

    fig, axes = plt.subplots(1, 2)

    def animate(_i: int):
        axes[0].clear()
        axes[1].clear()

        axes[0].set_title('Activity absorption generation')
        axes[1].set_title('Output label')

        axes[0].imshow(activity[_i, 0], cmap="Blues_r", vmin=-1, vmax=1)
        axes[1].imshow(label[_i, 0], cmap="Blues_r", vmin=-1, vmax=1)

    anim = animation.FuncAnimation(fig, animate, frames=len(activity))
    plt.show()

    plt.close()


def get_axis(axes, x, y, n_x, n_h):
    """
    Gets the axis for position x, y
    :param axes: Axes
    :param x: X-position of the axis
    :param y: Y-position of the axis
    :param n_x: Width of the kernel
    :param n_h: Height of the kernel
    :return: Axis
    """
    return axes[x][y] if n_x > 0 and n_h > 0 else axes[x] if n_x > 0 else axes[y] if n_h > 0 else axes


def show_kernel_activity(tensor: th.Tensor):
    """
    Plots the activity of the kernels for each position
    :param tensor: Tensor with kernel activity like [batch, kernel_width, kernel_height, width, height]
    """
    tensor = tensor.detach().cpu().numpy()
    kernel_width = tensor.shape[1]
    kernel_height = tensor.shape[2]

    tensor = tensor[0]
    fig, axes = plt.subplots(kernel_width, kernel_height)
    for x in range(kernel_width):
        for y in range(kernel_height):
            axis = get_axis(axes, x, y, kernel_width, kernel_height)
            axis.imshow(tensor[x, y])

    plt.suptitle('Activity of kernel')
    plt.show()

    plt.close()


def show_animation(*input: Tuple[str, np.ndarray]):
    fig, axes = plt.subplots(1, len(input))

    def animate(_t: int):
        for i in range(len(input)):
            label, data = input[i]
            vmin = np.min(data)
            vmax = np.max(data)
            axis = axes[i] if len(input) > 1 else axes
            axis.clear()
            axis.imshow(data[_t], vmin=vmin, vmax=vmax)

            plt.title(label)

    anim = animation.FuncAnimation(fig, animate, frames=len(input[0][1]))
    plt.show()

    plt.close()


def show3d(*tensor: th.Tensor, filename=None, scale=0.5, trained=70, teacher_forcing=10):
    """
    Shows tensors next to each other in pyvista
    :param tensor: Tensors to be plotted with pyvista
    :param filename: Name of the video of the plot
    :param scale: Scales the height
    :param trained: Number of time steps used for the training
    :param teacher_forcing: Number of time steps used for the teacher forcing
    """
    assert(len(tensor) > 0)
    assert(trained >= teacher_forcing)

    import pyvista as pv

    plotter = pv.Plotter()
    plotter.set_background("royalblue", top="aliceblue")
    title = plotter.add_text('0: teacher forcing')

    scale *= max(tensor[0].shape[-2:])

    t_steps = tensor[0].shape[1]
    
    # Pre processing tensors
    tensor_list = list()
    for _tensor in tensor:
        _tensor = _tensor.detach().cpu().numpy()
        # _tensor = np.clip(_tensor, -1, 1)
        # Batch 0
        _tensor = _tensor[0]
        tensor_list.append(_tensor)

    mesh_list = list()
    pts_list = list()
    for i, _tensor in enumerate(tensor_list):
        width, height = _tensor.shape[-2:]

        # Make mesh grid for plotting
        x = np.arange(0, width) - width * i  # Offsets grids
        y = np.arange(0, height) + height * i  # Offsets grids
        x, y = np.meshgrid(x, y)
        z = _tensor[0, 0] * scale

        # Structured surface for pyvista
        grid = pv.StructuredGrid(x, y, z)
        plotter.add_mesh(grid, scalars=z.ravel(), clim=[-1, 1], smooth_shading=True, cmap="ocean")
        mesh_list.append(plotter.mesh)

        pts_list.append(grid.points.copy().astype(float))

    plotter.show(auto_close=False, interactive_update=True)

    # Saves a movie
    if filename is None:
        filename = "plot.mp4"
    plotter.open_movie(filename)

    for t in range(t_steps):
        for _tensor, mesh, pts in zip(tensor_list, mesh_list, pts_list):
            z = _tensor[t, 0] * scale

            pts[:, -1] = z.ravel()

            plotter.update_coordinates(pts, mesh=mesh, render=False)
            plotter.update_scalars(z.ravel(), mesh=mesh, render=False)

            mesh.compute_normals(cell_normals=False, inplace=True)

        if t < teacher_forcing:
            title.SetText(2, f'{t + 1}: teacher forcing')
        elif t < trained:
            title.SetText(2, f'{t + 1} trained time frame')
        elif t >= trained:
            title.SetText(2, f'{t + 1}: unseen time frame')

        plotter.render()
        plotter.write_frame()

    #plotter.close()


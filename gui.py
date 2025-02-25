import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from skimage.draw import line_aa, disk
import config


def get_tk_im(array):
    array_im = np.asarray(array * 255, dtype='uint8')
    pilim = Image.fromarray(array_im)
    img = ImageTk.PhotoImage(image=pilim)
    return img


def draw_line(array_pos, array_neg, coord_start, coord_end, weight):
    """
    Draws a line on a np array.
    :param array: ndarray
        Image array to draw a line on.
    :param coord_start: (int, int)
        x and y position of start of line.
    :param coord_end: (int, int)
        x and y position of end of line.
    :return: ndarray
        Original image array with line drawn.
    """
    x1, y1 = coord_start
    x2, y2 = coord_end
    assert array_pos.shape[0] > x1 and array_pos.shape[0] > x2 and array_pos.shape[1] > y1 \
           and array_pos.shape[1] > y2, 'coordinates not in range '
    assert x1 >= 0 and x2 >= 0 and y1 >= 0 and y2 >= 0, 'Coordinates should be non-negative'
    assert array_pos.shape == array_neg.shape
    rr, cc, val = line_aa(x1, y1, x2, y2)
    if weight >= 0:
        array_pos[rr, cc] = 1 - val * weight * 2
    else:
        array_neg[rr, cc] = 1 - val * -weight * 2
    return array_pos, array_neg


class DistDelayGUI(object):
    """
    Animation for the distance-based delay network
    """

    def __init__(self, dist_delay_net):
        self.DDN = dist_delay_net
        self.spacing = 1/config.scale
        self.dot_size = 5
        x_range = self.DDN.x_range
        y_range = self.DDN.y_range
        width = x_range[1] - x_range[0]
        height = y_range[1] - y_range[0]
        self.w = int(width * self.spacing + self.dot_size * 2 + 1)
        self.h = int(height * self.spacing + self.dot_size * 2 + 1)
        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root, width=self.h + 50, height=self.w + 50)
        self.canvas.grid(row=0, column=0)
        self.connections_pos = None
        self.connections_neg = None
        self.get_connection_base()
        dot_ex, dot_in = self.grid2dots(True)
        all_d = np.stack([dot_in + dot_ex, dot_in + dot_ex, dot_in + dot_ex], axis=-1)
        connections_all = self.connections_pos + self.connections_neg
        self.img_c = np.stack([self.connections_pos, connections_all, self.connections_neg], axis=-1) * \
                     (all_d == 0)

    def get_connection_base(self):
        """
        Generates an image of the non-zero connections in this network.
        :return: None
        """
        weights = self.DDN.W
        # weights = np.asarray(weights > 0, dtype='uint8')
        grid = self.DDN.coordinates
        N = grid.shape[0]
        connections_pos = np.ones((self.w, self.h))
        connections_neg = np.ones((self.w, self.h))
        for n1 in range(N):
            for n2 in range(N):
                if weights[n1, n2] != 0:
                    x1 = int(grid[n1, 0] * self.spacing + self.dot_size)
                    y1 = int(grid[n1, 1] * self.spacing + self.dot_size)
                    x2 = int(grid[n2, 0] * self.spacing + self.dot_size)
                    y2 = int(grid[n2, 1] * self.spacing + self.dot_size)
                    connections_pos, connections_neg = draw_line(connections_pos, connections_neg, (x1, y1), (x2, y2), weights[n1, n2])
                    # xa1 = x2 + 0.95 * (x1 - x2) + 5
                    # ya1 = y2 + 0.95 * (y1 - y2) + 5
                    # xa2 = x2 + 0.95 * (x1 - x2) - 5
                    # ya2 = y2 + 0.95 * (y1 - y2) - 5
                    # connections = draw_line(connections, (int(xa1), int(ya1)), (x2, y2))
                    # connections = draw_line(connections, (int(xa2), int(ya2)), (x2, y2))
        self.connections_pos = connections_pos
        self.connections_neg = connections_neg

    def grid2dots(self, init=False):
        """
        Generates an image as a ndarray with neuron activation drawn as dots according to coordinates attribute.
        :param init: bool
            Sets activity of all neurons to 1. Only used when initial connection base visualisation is being
            created.
        :return: ndarray, ndarray
            Two w by h array with excitatory and inhibitory neuron activation respectively as dots drawn in the
            array.
        """
        coordinates = self.DDN.coordinates
        act = self.DDN.A
        if init:
            act = np.ones_like(act)
        n_type = self.DDN.n_type
        N = coordinates.shape[0]
        dots_ex = np.zeros((self.w, self.h))
        dots_in = np.zeros((self.w, self.h))
        for i in range(N):
            a = act[i]
            type = n_type[i]
            x = np.round(coordinates[i, 0] * self.spacing + self.dot_size)
            y = np.round(coordinates[i, 1] * self.spacing + self.dot_size)
            rr, cc = disk((x, y), self.dot_size)
            # if type == 1:
            #     dots_ex[rr, cc] = a
            # else:
            #     dots_in[rr, cc] = a
            if a > 0:
                dots_ex[rr, cc] = a
                dots_in[rr, cc] = 0
            else:
                dots_ex[rr, cc] = 0
                dots_in[rr, cc] = -a
        return dots_ex, dots_in

    def update_a(self):
        """
        Updates the gui according to current neuron activity.
        :return: None
        """
        dot_ex, dot_in = self.grid2dots()
        zer = np.zeros_like(self.connections_pos)
        img_d = np.stack([dot_in, dot_ex, zer], axis=-1)
        img = self.img_c + img_d
        img = get_tk_im(img)

        self.canvas.create_image(10, 40, anchor="nw", image=img)
        self.root.update()

    def close(self):
        self.root.destroy()

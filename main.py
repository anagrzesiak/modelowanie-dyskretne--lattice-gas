import tkinter as tk
import random
import numpy as np
import numba
from PIL import Image, ImageTk
from numba import jit


@jit(nopython=True, parallel=True)
def original_states(image_matrix, gas_state_matrix):
    for x in numba.prange(0, gas_state_matrix.shape[0]):
        for y in numba.prange(0, gas_state_matrix.shape[1]):
            state = gas_state_matrix[x, y]
            if y == 1 or x == 1 or y == gas_state_matrix.shape[1] - 2 or x == gas_state_matrix.shape[0] - 2:
                state[0] = 1
                gas_state_matrix[x, y] = state
                image_matrix[x, y] = [255, 255, 255]
            if y == 100 and (
                    0 < x < (((gas_state_matrix.shape[0]) / 2) - 25) or ((gas_state_matrix.shape[0]) / 2) + 25 < x <
                    gas_state_matrix.shape[0]):
                state[0] = 1
                gas_state_matrix[x, y] = state
                image_matrix[x, y] = [255, 255, 255]
            generator = random.randint(0, 100)
            bound = 90
            if 0 < y < 100 and 0 < x < (gas_state_matrix.shape[0] - 1) and generator > bound:
                state_generator = random.randint(1, 4)
                if state_generator == 1:
                    state[1] = 1
                elif state_generator == 2:
                    state[2] = 1
                elif state_generator == 3:
                    state[3] = 1
                elif state_generator == 4:
                    state[4] = 1
                gas_state_matrix[x, y] = state
                image_matrix[x, y] = [255, 0, 0]
    return image_matrix, gas_state_matrix


@jit(nopython=True, parallel=True)
def update_states(gas_state_matrix, image_matrix, ok_gas_state_matrix, ok_image_matrix):
    for x in numba.prange(0, gas_state_matrix.shape[0] - 1):
        for y in numba.prange(0, gas_state_matrix.shape[1] - 1):
            state = gas_state_matrix[x, y]
            if state[0] == 1:
                ok_gas_state_matrix[x, y] = [1, 0, 0, 0, 0]
                ok_image_matrix[x, y] = [255, 255, 255]
            elif state[0] == 0:
                ok_state = [0, 0, 0, 0, 0]
                north_state = gas_state_matrix[x - 1, y]
                east_state = gas_state_matrix[x, y + 1]
                south_state = gas_state_matrix[x + 1, y]
                west_state = gas_state_matrix[x, y - 1]
                if state[1] == 1 and north_state[0] == 1:
                    ok_state[1] = 0
                    ok_state[3] = 1
                elif state[2] == 1 and east_state[0] == 1:
                    ok_state[2] = 0
                    ok_state[4] = 1
                elif state[3] == 1 and south_state[0] == 1:
                    ok_state[1] = 1
                    ok_state[3] = 0
                elif state[4] == 1 and west_state[0] == 1:
                    ok_state[2] = 1
                    ok_state[4] = 0
                else:
                    if south_state[1] == 1:
                        ok_state[1] = 1
                        south_state[1] = 0
                    if west_state[2] == 1:
                        ok_state[2] = 1
                        west_state[2] = 0
                    if north_state[3] == 1:
                        ok_state[3] = 1
                        north_state[3] = 0
                    if east_state[4] == 1:
                        ok_state[4] = 1
                        east_state[4] = 0
                    if ok_state[1] == 1 and ok_state[3] == 1:
                        ok_state[1] = 0
                        ok_state[2] = 1
                        ok_state[3] = 0
                        ok_state[4] = 1
                    elif ok_state[2] == 1 and ok_state[4] == 1:
                        ok_state[1] = 1
                        ok_state[2] = 0
                        ok_state[3] = 1
                        ok_state[4] = 0
                ok_gas_state_matrix[x, y] = ok_state
                if ok_state[1] == 1 or ok_state[2] == 1 or ok_state[3] == 1 or ok_state[4] == 1:
                    ok_image_matrix[x, y] = [232, 64, 170]

    return ok_gas_state_matrix, ok_image_matrix


def animate(master, frame, canvas, image_matrix, gas_state_matrix):
    ok_gas_state_matrix = np.zeros([gas_state_matrix.shape[0], gas_state_matrix.shape[1], 5], dtype=np.uint8)
    ok_image_matrix = np.zeros([image_matrix.shape[0], image_matrix.shape[1], 3], dtype=np.uint8)
    ok_gas_state_matrix, ok_image_matrix = update_states(gas_state_matrix, image_matrix, ok_gas_state_matrix,
                                                         ok_image_matrix)
    img = ImageTk.PhotoImage(image=Image.fromarray(ok_image_matrix))
    canvas.image = img
    canvas.create_image(0, 0, anchor="nw", image=img)
    master.after(10, lambda: animate(master, frame, canvas, ok_image_matrix, ok_gas_state_matrix))


class GUI:
    def __init__(self, master):
        self.master = master
        self.frame = tk.Frame(self.master, background="black")
        self.image_matrix = np.zeros([600, 600, 3], dtype=np.uint8)
        self.gas_state_matrix = np.zeros([600, 600, 5], dtype=np.uint8)
        self.image_matrix, self.gas_state_matrix = original_states(self.image_matrix, self.gas_state_matrix)
        self.img = ImageTk.PhotoImage(image=Image.fromarray(self.image_matrix))
        self.canvas = tk.Canvas(self.frame, width=600, height=600)
        self.canvas.pack(side="left", fill="y")
        self.canvas.create_image(0, 0, anchor="nw", image=self.img)
        self.frame.pack()
        self.master.after(50, lambda: animate(self.master, self.frame, self.canvas, self.image_matrix,
                                              self.gas_state_matrix))


def main():
    root = tk.Tk()
    root.withdraw()
    ok = tk.Toplevel(root)
    ok.protocol("WM_DELETE_WINDOW", root.destroy)
    app = GUI(ok)
    ok.title("XDDDDD")
    ok.mainloop()


if __name__ == '__main__':
    main()

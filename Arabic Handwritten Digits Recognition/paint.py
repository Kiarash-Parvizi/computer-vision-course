import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

import matplotlib.pyplot as plt

# Load model from file
model = models.load_model('storage/cnn001.h5')


def show_img(img) -> None:
    for row in img:
        for v in row:
            if v[0] != 0:
                print(1, end='')
            else:
                print(0, end='')
        print('')
    print('-'*20)

def oneHot2Label(pred) -> int:
    pred = pred[0]
    mx, mxV = 0, pred[0]
    for i in range(1,len(pred)):
        if pred[i] > mxV: mx, mxV = i, pred[i]
    return mx

def fit_center(img) -> np.ndarray:
    extraStartRow, extraEndRow = 0, 0
    # find the extra startRow
    for i in range(img.shape[0]):
        if np.any(img[i]): break
        extraStartRow += 1
    # find the extra endRow
    for i in range(img.shape[0]-1, -1, -1):
        if np.any(img[i]): break
        extraEndRow += 1
    # do the same for columns
    extraStartCol, extraEndCol = 0, 0
    # find the extra startCol
    for i in range(img.shape[1]):
        if np.any(img[:,i]): break
        extraStartCol += 1
    # find the extra endCol
    for i in range(img.shape[1]-1, -1, -1):
        if np.any(img[:,i]): break
        extraEndCol += 1
    # return
    return np.roll(img, shift=((extraEndCol-extraStartCol)//2, (extraEndRow-extraStartRow)//2), axis=(1, 0))

import tkinter as tk

from scipy.ndimage.filters import gaussian_filter

class DrawableGrid(tk.Frame):
    def __init__(self, parent, width, height, size=5):
        super().__init__(parent, bd=1, relief="sunken")
        self.width = width
        self.height = height
        self.size = size
        canvas_width = width*size
        canvas_height = height*size
        self.canvas = tk.Canvas(self, bd=0, highlightthickness=0, width=canvas_width, height=canvas_height)
        self.canvas.pack(fill="both", expand=True, padx=2, pady=2)

        for row in range(self.height):
            for column in range(self.width):
                x0, y0 = (column * size), (row*size)
                x1, y1 = (x0 + size), (y0 + size)
                self.canvas.create_rectangle(x0, y0, x1, y1,
                                             fill="white", outline="gray",
                                             tags=(self._tag(row, column),"cell" ))
        self.canvas.tag_bind("cell", "<B1-Motion>", self.paint)
        self.canvas.tag_bind("cell", "<1>", self.paint)
        self.canvas.bind("<ButtonRelease>", self.Done)

    def _tag(self, row, column):
        """Return the tag for a given row and column"""
        tag = f"{row},{column}"
        return tag

    def print_pixels(self):
        row = ""
        for row in range(self.height):
            output = ""
            for column in range(self.width):
                color = self.canvas.itemcget(self._tag(row, column), "fill")
                value = "1" if color == "black" else "0"
                output += value
            print(output)
    def get_pixels(self):
        res = []
        for row in range(self.height):
            ls = []
            for column in range(self.width):
                color = self.canvas.itemcget(self._tag(row, column), "fill")
                value = 255 if color == "black" else 0
                ls.append(value)
            res.append(ls)
        return res
    def get_pixels_flatten(self):
        res = []
        for row in range(self.height):
            for column in range(self.width):
                color = self.canvas.itemcget(self._tag(row, column), "fill")
                value = 255 if color == "black" else 0
                res.append(value)
        return res

    def clear(self):
        for row in range(self.height):
            for column in range(self.width):
                cell = self.canvas.gettags(self._tag(row, column))[0]
                self.canvas.itemconfigure(cell, fill="white")

    def paint(self, event):
        cell = self.canvas.find_closest(event.x, event.y)
        self.canvas.itemconfigure(cell, fill="black")

    def Done(self, event):
        data = np.array([self.get_pixels_flatten()])
        # transformation
        data = data.reshape(-1,28,28,1)
        #data[0] = np.rot90(data[0].reshape(28,28), k=1, axes=(0, 1)).reshape(-1)
        #data[0] = np.flip(data[0].reshape(28,28), 0).reshape(-1)
        # blur
        data[0] = fit_center(data[0])
        data[0] = gaussian_filter(data[0], sigma=0.5)
        data = data.astype('float32') / 255.0
        # ann
        pred = model.predict(data)
        print(pred)
        print(f'number: {oneHot2Label(pred)}')
        plt.figure()
        plt.imshow(data.reshape(-1,28,28)[0])
        plt.show()
        #
        self.clear()


root = tk.Tk()

canvas = DrawableGrid(root, width=28, height=28, size=10)
b = tk.Button(root, text="Print Data", command=canvas.print_pixels)
b.pack(side="top")
canvas.pack(fill="both", expand=True)
root.mainloop()


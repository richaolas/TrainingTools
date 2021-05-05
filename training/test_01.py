# -*- coding: utf-8 -*-

import numpy as np
import cv2


def gaussian_label(sz, sigma):
    w, h = sz
    xs, ys = np.meshgrid(np.arange(w) - w // 2, np.arange(h) - h // 2)
    labels = np.exp(-0.5 * (xs ** 2 + ys ** 2) / (sigma ** 2))
    labels = np.roll(labels, -int(np.floor(w / 2)), axis=1)
    labels = np.roll(labels, -int(np.floor(h / 2)), axis=0)
    return labels


def get_cos_window(sz):
    w, h = sz
    cos_window = np.hanning(h)[:, np.newaxis].dot(np.hanning(w)[np.newaxis, :])
    return cos_window

class App(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        ret, self.frame = self.cap.read()
        cv2.namedWindow('CFtracking')
        cv2.setMouseCallback('CFtracking', self.on_mouse)
        self.track_window = None
        self.selection = None
        self.drag_start = None
        self.tracking = False

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
        if event == cv2.EVENT_LBUTTONUP:
            self.drag_start = None
            self.track_window = self.selection
        if self.drag_start and event == cv2.EVENT_MOUSEMOVE:
            xmin = min(x, self.drag_start[0])
            ymin = min(y, self.drag_start[1])
            xmax = max(x, self.drag_start[0])
            ymax = max(y, self.drag_start[1])
            self.selection = (xmin, ymin, xmax - xmin, ymax - ymin)
            print(self.selection)

    def run(self):
        while True:
            ret, self.frame = self.cap.read()
            show = self.frame.copy()
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

            try:
                if self.track_window:
                    x, y, w, h = self.track_window
                    self.target_sz = (w, h)
                    self.window_sz = (int((1+1.5)*w), int((1+1.5)*h))
                    self.center = (x+w/2,y+h/2)
                    label = gaussian_label(self.window_sz, np.sqrt(w * h) // 10)
                    self.yf = np.fft.fft2(label)
                    self.cos_window = get_cos_window(self.window_sz)
                    xt = cv2.getRectSubPix(gray, self.window_sz, self.center) / 255 - 0.5
                    xt = xt * self.cos_window
                    xf = np.fft.fft2(xt)
                    hf = np.conjugate(self.yf) * xf / (np.conjugate(xf) * xf + 1e-3)
                    self.track_window = None
                    self.tracking = True

                if self.tracking:
                    xt = cv2.getRectSubPix(gray, self.window_sz, self.center) / 255 - 0.5
                    xt = xt * self.cos_window
                    xf = np.fft.fft2(xt)
                    response = np.real(np.fft.ifft2(np.conjugate(hf) * xf))
                    dy, dx = np.unravel_index(np.argmax(response, axis=None), response.shape)
                    if dx + 1 > self.window_sz[0]/2:
                        dx = dx - self.window_sz[0]
                    if dy + 1 > self.window_sz[1]/2:
                        dy = dy - self.window_sz[1]
                    xc,yc = self.center
                    xc += dx
                    yc += dy
                    self.center = (xc,yc)
                    xt = cv2.getRectSubPix(gray, self.window_sz, self.center) / 255 - 0.5
                    xt = xt * self.cos_window
                    xf = np.fft.fft2(xt)
                    new_hf = np.conjugate(self.yf) * xf / (np.conjugate(xf) * xf + 1e-3)
                    hf = (1 - 0.075) * hf + 0.075 * new_hf
                    xc, yc = self.center
                    print((xc - w/2,yc - h/2), (xc + w/2, yc + h/2))
                    #cv2.rectangle(show, (xc - w/2,yc - h/2), (xc + w/2, yc + h/2), (0, 0, 255), 2)
            except:
                break

            cv2.imshow('CFtracking', show)
            if cv2.waitKey(1) == 27:
                break
        print('Program terminate')
        cv2.destroyAllWindows()


if __name__ == '__main__':
    app = App()
    app.run()

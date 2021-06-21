# Script to get images from BLACS and process them for absorption imaging of
# atoms, using background subtraction and linear reconstruction of probe
# images
import sys
import json
import threading
from pathlib import Path

import numpy as np
from tqdm import tqdm
import zmq
from qtutils import inmain, inmain_decorator
from qtutils.qt import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

from image_reconstruction.image_averager import ImageAverager
from image_reconstruction.cpu_reconstructor import CPUReconstructor

try:
    PUB_PORT = int(sys.argv[1])
except (ValueError, IndexError):
    raise ValueError("Must provide camera's BLACS tab's PUB port number as a command line arg.")


SAVE_RAW_IMAGES = True
LOAD_SAVED_IMAGES = True

if SAVE_RAW_IMAGES:
    Path('saved_images').mkdirs(exist_ok=True)

class ReconstructedAbsorptionImaging:
    # Number of background images 
    N_bg = 100

    # Number of probe images used for reconstruction basis:
    N_probe = 100

    def __init__(self):
        self.averager = ImageAverager(self.N_bg)
        self.reconstructor = CPUReconstructor(self.N_probe)
        self.ctx = zmq.Context()
        self.sock = None
        self.dark = None
        self.image = None
        self.image_rendering_required = threading.Event()
        self.image_view = None
        self.stopping = False
        self.roi = None
        self.mask_slice = None
        self.mask = None
        self.first_image_shown = False

    def new_sock(self):
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.SUBSCRIBE, b'')
        self.sock.setsockopt(zmq.RCVHWM, 1)
        self.sock.connect(f'tcp://localhost:{PUB_PORT}')

    def recvimg(self):
        data = self.sock.recv_multipart()
        md = json.loads(data[0])
        image = np.frombuffer(memoryview(data[1]), dtype=md['dtype'])
        image = image.reshape(md['shape'])
        return image

    def collect_bg_images(self):
        input("Begin acquiring dark images and press enter")
        self.new_sock()
        for i in tqdm(range(self.N_bg), desc="collecting dark images"):
            if LOAD_SAVED_IMAGES:
                image = np.load(f'saved_images/dark_{i:04d}.npy')
            else:
                image = self.recvimg()
            if SAVE_RAW_IMAGES:
                np.save(f'saved_images/dark_{i:04d}.npy', image)
            self.averager.add_ref_image(image)
            
        self.sock.close()
        self.dark = self.averager.get_average()

    def collect_probe_images(self):
        input("Begin acquiring probe images and press enter")
        self.new_sock()
        for i in tqdm(range(self.N_probe), desc="collecting probe images"):
            if LOAD_SAVED_IMAGES:
                image = np.load(f'saved_images/probe_{i:04d}.npy')
            else:
                image = self.recvimg()
            if SAVE_RAW_IMAGES:
                np.save(f'saved_images/probe_{i:04d}.npy', image)
            self.reconstructor.add_ref_image(image)
        self.sock.close()

    def start_absorption_images(self):
        input("Begin acquiring absorption images and press enter")
        thread = threading.Thread(target=self._collect_absorption_images, daemon=True)
        thread.start()
        app = QtWidgets.QApplication(sys.argv)
        render_timer = QtCore.QTimer()
        render_timer.timeout.connect(self.render_image)
        render_timer.start(20)
        self.image_view = pg.ImageView()
        self.roi = pg.RectROI([100, 100], [100, 100], pen=(0,9))
        self.image_view.addItem(self.roi)
        self.image_view.show()
        app.exec()
        self.stopping = True
        thread.join()

    def _collect_absorption_images(self):
        self.new_sock()
        for i in range(self.N_probe):
            if LOAD_SAVED_IMAGES:
                absorption = np.load(f'saved_images/atoms_{i:04d}.npy')
            else:
                absorption = self.recvimg()
            if SAVE_RAW_IMAGES:
                np.save(f'saved_images/probe_{i:04d}.npy', absorption)
            recon_probe, _ = self.reconstructor.reconstruct(absorption, mask=self.mask)
            absorbed_fraction = 1 - (absorption.astype(float) - self.dark) / (recon_probe - self.dark)
            self.image = 1000 * absorbed_fraction
            self.image_rendering_required.set()
        self.sock.close()

    def render_image(self):
        if self.image_rendering_required.is_set():
            self.image_rendering_required.clear()
            if not self.first_image_shown:
                self.image_view.setImage(
                    self.image.swapaxes(-1, -2),
                    autoRange=True,
                    levels=(0,1000)
                )
                self.image_view.setHistogramRange(-1, 2)
                self.first_image_shown = True
            else:
                self.image_view.setImage(
                    self.image.swapaxes(-1, -2),
                    autoRange=False,
                    autoLevels=False,
                    autoHistogramRange=False,
                )
            QtGui.QApplication.instance().sendPostedEvents()
            # Update mask if need be:
            image_item = self.image_view.getImageItem()
            mask_slice, _ = self.roi.getArraySlice(self.image, image_item, axes=(1, 0))
            if mask_slice != self.mask_slice:
                self.mask_slice = mask_slice
                mask = np.ones(self.image.shape, dtype=bool)
                mask[mask_slice] = 0
                self.mask = mask

imaging = ReconstructedAbsorptionImaging()
imaging.collect_bg_images()
imaging.collect_probe_images()
imaging.start_absorption_images()

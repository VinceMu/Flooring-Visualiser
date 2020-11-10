import numpy as np
"""
Input image boundaries for image detection models
"""
class ImageSpec:
    def __init__(self,
                 height: float,
                 width: float,
                 channel: int = 3,
                 batch_size: int = 1):
        self.height = height
        self.width = width
        self.channels = channel
        self.batch_size = batch_size

    def check_image(self, image: np.ndarray):
        if (self.batch_size, self.height, self.width,
                self.channels) != image.shape:
            return False
        else:
            return True

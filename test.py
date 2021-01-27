import cv2, numpy as np
from PIL import Image
from services.floor_visualiser import FloorVisualiser
from services.texture_synthesizer import TextureSynthesizer
from model import QuiltMode

FLOOR_RGB_COLOR = (80, 50, 50)

src = cv2.imread("data/segmentation_test_set/living_space.jpg")
img = cv2.imread("data/segmentation_results/ade20k-resnet101dilated-ppm_deepsup/living_space.jpg")
texture = cv2.imread("data/texture_synthesis_results/good1.jpg")

visualiser = FloorVisualiser(TextureSynthesizer(1.2))
# visualiser.boundSegmentedArea(segmentedFloorRaw)
res = visualiser.replaceFloor(src,img,texture)

Image.fromarray(res).show()

# sample_texture = cv2.cvtColor(cv2.imread("data/texture_synthesis_test_set/wooden_floor_board.jpg"),cv2.COLOR_BGR2RGB)
# synthesizer = TextureSynthesizer(2)
# synthesized_texture = synthesizer.synthesizeTexture(sample_texture,(2048,1997))
# Image.fromarray(synthesized_texture).save("data/tmp.jpg")

# CUT = 60 x 7 - 60
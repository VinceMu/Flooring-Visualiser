import cv2
import numpy as np
import pandas as pd
from PIL import Image
import os

# transform category here
# filename = "../data/ADEChallengeData2016/annotations/training/ADE_train_00000001.png"
# Image.fromarray(masked.astype(np.uint8)).show()


train_directory = "../data/ADEChallengeData2016/annotations/training"
new_train_directory = "../data/ADEChallengeData2016Masked/annotations/training"
i = 0
for filename in os.listdir(train_directory):
    if filename.endswith(".png"):
        print(i)
        path = os.path.join(train_directory, filename)
        img = cv2.imread(path)
        masked = np.where(img == 4, 1, 0).astype(np.uint8)

        new_path = os.path.join(new_train_directory, filename)
        cv2.imwrite(new_path, masked)
        i += 1
    else:
        continue

train_directory = "../data/ADEChallengeData2016/annotations/validation"
new_train_directory = "../data/ADEChallengeData2016Masked/annotations/validation"
i = 0
for filename in os.listdir(train_directory):
    if filename.endswith(".png"):
        print(i)
        path = os.path.join(train_directory, filename)
        img = cv2.imread(path)
        masked = np.where(img == 4, 1, 0).astype(np.uint8)

        new_path = os.path.join(new_train_directory, filename)
        cv2.imwrite(new_path, masked)
        i += 1
    else:
        continue

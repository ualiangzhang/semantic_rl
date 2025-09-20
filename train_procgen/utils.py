import argparse
from collections import OrderedDict
import numpy as np
import os
import cv2
from os.path import isfile, join


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def convert_frames_to_video(pathIn, pathOut, fps=8, limits=480):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f)) and (
            join(pathIn, f).endswith(".jpg") or join(pathIn, f).endswith(".jpeg") or join(pathIn, f).endswith(
        "png"))]
    if len(files) > limits:
        files = files[:limits]
    if len(files) == 0:
        return
    # for sorting the file names properly
    files.sort(key=lambda x: int(x.split("_")[0]))
    for i in range(len(files)):
        filename = pathIn + files[i]
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith("png"):
            # reading each files
            img = cv2.imread(filename)
            # resize the image for saving space
            img_resized = cv2.resize(img, (256, 256))
            height, width, layers = img_resized.shape
            size = (width, height)
            # inserting the frames into an image array
            frame_array.append(img_resized)

    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

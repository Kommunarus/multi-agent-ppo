import os
import imageio

import cv2
path = '/home/neptun/Изображения/hak/'
all_img_svg = sorted(os.listdir(path), key=lambda x: int(x.replace('.png','')))



frames = []
for i, file in enumerate(all_img_svg):
    frame = cv2.imread(path+file)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (433, 394))
    frames.append(frame)

print("Saving GIF file")
with imageio.get_writer("robo.gif", mode="I") as writer:
    for idx, frame in enumerate(frames):
        print("Adding frame to GIF file: ", idx + 1)
        writer.append_data(frame)
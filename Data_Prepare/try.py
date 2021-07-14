import numpy as np
from PIL import Image
import os
from clip_and_resize import *


src_folder = '/home/user1/Documents/css-unet-main/UNET1/diamonds_labels_cutted'
save_folder = '/home/user1/Documents/css-unet-main/tmp'


f = open("/home/user1/Documents/css-unet-main/UNET1/id.txt","r")

for line in f:
	filename = line[:-1]
	label_filename = filename[:11] + "-mask.png"
	try:
		handle_image(filename, label_filename, src_folder=src_folder, save_folder=save_folder)
		print(label_filename)
	except IOError:
		pass

f.close()

#file = open(os.path.join(save_folder,'cutted.txt'),'a')

#Image.open(os.path.join(src_folder, filename))



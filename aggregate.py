import joblib
import cv2
import image_process
from parameters import *
import copy
import numpy as np
import genetics


image_process.load('images_serialized/images_with_info.dict')
individual = joblib.load('transformations/best.ind')
for i, (name, info) in enumerate(image_process.IMAGES_WITH_INFO.items()):
    if i == TRAIN_NO: break

    image, gray_image, mask, no_cells = info
    processed = image_process.process_image(copy.copy(image), individual)
    background = image_process.kmeans(processed, 1)
    if background > np.array(128):
        processed = (255 - processed)

    cv2.imwrite('results2/'+name+'_image.png', image)
    cv2.imwrite('results2/'+name+'_mask.png', mask)
    cv2.imwrite('results2/'+name+'_result.png', processed)
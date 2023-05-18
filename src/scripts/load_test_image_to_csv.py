import os 
from PIL import Image
import numpy as np
import pandas as pd

dirname = '../../data/datasets/test_set/'
rows_list = []
for fname in os.listdir(dirname):
    dict1 = {}
    try:
        im = Image.open(os.path.join(dirname, fname))
    except:
        continue
    imarray = np.array(im)
    dict1['file'] = fname
    dict1['image'] = imarray
    rows_list.append(dict1)
df = pd.DataFrame(rows_list, columns=['file', 'image'])
df.to_csv(f'{dirname}df_images_test.csv')
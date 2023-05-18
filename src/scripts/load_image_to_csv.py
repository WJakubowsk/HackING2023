import os 
from PIL import Image
import numpy as np
import pandas as pd

dirname = '../../data/datasets/train_set/'
df = pd.DataFrame()
for dirdirname in os.listdir(dirname):
    new_path = os.path.join(dirname, dirdirname) + "/"
    rows_list = []
    for fname in os.listdir(new_path):
        dict1 = {}
        im = Image.open(os.path.join(new_path, fname))
        imarray = np.array(im)
        dict1['file'] = fname
        dict1['image'] = imarray
        rows_list.append(dict1)
    df1 = pd.DataFrame(rows_list, columns=['file', 'image'])
    df1['label'] = dirdirname 
    df = pd.concat([df, df1])
df.to_csv(f'{dirname}df_images.csv')
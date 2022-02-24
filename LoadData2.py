from PIL import Image
import pandas as pd
import os
import numpy as np

# df = pd.read_csv(r'C:\Users\user\Desktop\crop2_data\images')
df = pd.read_csv(r'/mnt/d/tr.csv')
rawImageAdress = r'/mnt/d/ntut/crop2_data/images'
maskImageAdress = r'/mnt/d/ntut/crop2_data/masks'

imageR = Image.open(os.path.join(rawImageAdress, df['ImageId'][0])).convert('RGB')
imageM = Image.open(os.path.join(maskImageAdress, df['MaskId'][15]))
imageM = Image.fromarray(np.array(imageM)[:,:,1]).convert('P')

# print(os.getcwd())
imageM.show()


#----------------------------------------------------------------------------------------------

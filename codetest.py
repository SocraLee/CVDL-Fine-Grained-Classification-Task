import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
image = cv2.imread('./train/00000.jpg')
print(image)
print(image.shape)
#print(cv2.shape)
# test_df = pd.read_csv('./result.csv')
# print(test_df.head(5))
# test_df=test_df[['ID','Category']]
# test_df.to_csv('./result1.csv',index=False)

plt.figure()
plt.imshow(image)
plt.show()


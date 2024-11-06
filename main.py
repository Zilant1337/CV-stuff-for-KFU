import numpy as np
import PIL
from matplotlib import pyplot as plt

a = 1
# Импорт в массив
with PIL.Image.open("KOS-MOS.jpg") as img:
    a = np.array(img)
plt.imshow(a)
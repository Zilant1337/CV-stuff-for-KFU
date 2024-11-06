import numpy
import numpy as np
import PIL
from matplotlib import pyplot as plt

a = np.array(0)

# Импорт в массив
with PIL.Image.open("helldiver.jpg") as img:
    a = np.array(img)
plt.imshow(a)
plt.show()

# Inverting colour
invert_tensor = np.full(a.shape,255)
a= invert_tensor-a
plt.imshow(a)
plt.show()

# Grayscaling the image

import numpy
import numpy as np
import PIL
from matplotlib import pyplot as plt

ntsc_values = [0.299, 0.587, 0.114]

a = np.array(0)

# Импорт в массив
with PIL.Image.open("helldiver.jpg") as img:
    a = np.array(img)
plt.imshow(a)
plt.show()

# Inverting colour
# invert_tensor = np.full(a.shape,255)
# a= invert_tensor-a
# plt.imshow(a)
# plt.show()

# Greyscaling the image
grey_a = np.full((a.shape[0],a.shape[1]),0, dtype=np.float64)
for i in range (3):
    grey_a+= np.multiply(a[:,:,i],ntsc_values[i])

plt.imshow(grey_a,cmap='gray')
plt.show()




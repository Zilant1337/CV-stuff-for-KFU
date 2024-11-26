import math
import numpy
import numpy as np
import PIL
from matplotlib import pyplot as plt

ntsc_values = [0.299, 0.587, 0.114]
# print("Please input the size of the kernel:")
# kernel_size_input = int(input())
# print("Please input the gaussian standart deviation:")
# gaussian_standart_deviation = int(input())
# gaussian_kernel_size = kernel_size_input - (1 - kernel_size_input % 2)
# a = np.array(0)
#
# # Creating kernel
# kernel = np.zeros((gaussian_kernel_size,gaussian_kernel_size))
# for x in range(0, gaussian_kernel_size):
#     for y in range(0, gaussian_kernel_size):
#         x_from_center = gaussian_kernel_size//2+1-x
#         y_from_center = gaussian_kernel_size // 2 + 1 - y
#         kernel[x,y]=1/(2*math.pi*gaussian_standart_deviation**2)*math.exp(-1*(x_from_center**2+y_from_center**2)/(2*gaussian_standart_deviation**2))


# # Importing image into a list
# with PIL.Image.open("helldiver.jpg") as img:
#     a = np.array(img)
# plt.imshow(a)
# plt.show()
#
#
# # Inverting colour
# invert_tensor = np.full(a.shape,255)
# a= invert_tensor-a
# plt.imshow(a)
# plt.show()
#
# # Greyscaling the image
# grey_a = np.full((a.shape[0],a.shape[1]),0, dtype=np.float64)
# for i in range (3):
#     grey_a+= np.multiply(a[:,:,i],ntsc_values[i])
#
# plt.imshow(grey_a,cmap='gray')
# plt.show()
#
# hist = np.histogram(grey_a,bins=range(0,255))
# plt.bar(x=hist[1][1:], height = hist[0])
# plt.show()
#
# # Adding random noise
# noise_matrix = np.random.normal(loc = 50,scale=50,size=grey_a.shape)
# grey_a+=noise_matrix
# plt.imshow(grey_a,cmap='gray')
# plt.show()
#
# hist = np.histogram(grey_a,bins=range(0,255))
# plt.bar(x=hist[1][1:], height = hist[0])
# plt.show()
#
# # Gaussian filtering
#
# bordered_shape = [x + gaussian_kernel_size * 2 for x in grey_a.shape]
# bordered_gray_a = np.full(bordered_shape,0)
# bordered_gray_a[gaussian_kernel_size:gaussian_kernel_size + grey_a.shape[0], gaussian_kernel_size:gaussian_kernel_size + grey_a.shape[1]] = grey_a
#
# gaussian_blur_a = np.full(grey_a.shape,0)
# for i in range(gaussian_kernel_size,bordered_gray_a.shape[0]-gaussian_kernel_size):
#     for j in range(gaussian_kernel_size,bordered_gray_a.shape[1]-gaussian_kernel_size):
#         slice_for_kernel = bordered_gray_a[i-gaussian_kernel_size//2:i+gaussian_kernel_size//2+1,j-gaussian_kernel_size//2:j+gaussian_kernel_size//2+1]
#         slice_for_kernel = np.multiply(slice_for_kernel,kernel)
#         # print(f"X: {i-gaussian_kernel_size}, Y: {j-gaussian_kernel_size}")
#         gaussian_blur_a[i-gaussian_kernel_size, j-gaussian_kernel_size] = np.sum(slice_for_kernel)
#
#
# plt.imshow(gaussian_blur_a,cmap='gray')
# plt.show()
#
# hist = np.histogram(gaussian_blur_a,bins=range(0,255))
# plt.bar(x=hist[1][1:], height = hist[0])
# plt.show()
#
# # Equalizing the histogram
# total_img_pixel_count = a.shape[0]*a.shape[1]
#
#
# def cdf(list,i):
#     return np.sum(list[:i])
#
# cdf_min = cdf(hist[0],0)
#
# for x in range(len(gaussian_blur_a)):
#     for y in range(len(gaussian_blur_a[0])):
#         gaussian_blur_a[x,y] = round((cdf(hist[0],round(gaussian_blur_a[x,y]))-cdf_min)/(total_img_pixel_count-1)*255)
#
# plt.imshow(gaussian_blur_a,cmap='gray')
# plt.show()
#
# hist = np.histogram(gaussian_blur_a,bins=range(0,255))
# plt.bar(x=hist[1][1:], height = hist[0])
# plt.show()

# Otsu binarization
b = np.array(0)
c = np.array(0)
with PIL.Image.open("tarkov2.jpeg") as img:
    b = np.array(img)
with PIL.Image.open("tarkov3.jpeg") as img:
    c = np.array(img)

grey_b = np.full((b.shape[0],b.shape[1]),0, dtype=np.float64)
grey_c = np.full((c.shape[0],c.shape[1]),0, dtype=np.float64)
for i in range (3):
    grey_b+= np.multiply(b[:,:,i],ntsc_values[i])
    grey_c += np.multiply(c[:, :, i], ntsc_values[i])
plt.imshow(grey_b, cmap = "grey")
plt.show()
plt.imshow(grey_c, cmap = "grey")
plt.show()

def otsu_threshold(img):
    histogram = np.histogram(img,bins = range(0,257))
    total_pixels = len(img)*len(img[0])
    probability = histogram[0]/total_pixels
    cum_sum = np.cumsum(probability)

    cumulative_mean = np.cumsum(probability * np.arange(256))

    max_variance = 0
    threshold = 0

    for t in range(1, 256):
        q1 = cum_sum[t]
        q2 = 1 - q1

        mu1 = cumulative_mean[t] / q1 if q1 != 0 else 0
        mu2 = (cumulative_mean[-1] - cumulative_mean[t]) / q2 if q2 != 0 else 0

        variance_between = q1 * q2 * (mu1 - mu2) ** 2

        if variance_between > max_variance:
            max_variance = variance_between
            threshold = t

    return threshold

b_threshold = otsu_threshold(grey_b)
c_threshold = otsu_threshold(grey_c)

binary_b = np.full((b.shape[0],b.shape[1]),0, dtype=np.float64)
binary_c = np.full((c.shape[0],c.shape[1]),0, dtype=np.float64)
def binarization(img,threshold):
    binary_img = np.full((img.shape[0],img.shape[1]),0, dtype=np.float64)
    for x in range(len(img)):
        for y in range(len (img[0])):
            if(img[x,y]<threshold):
                binary_img[x,y]=0
            else:
                binary_img[x,y]=256
    return binary_img

binary_b = binarization(grey_b,b_threshold)
binary_c = binarization(grey_c,c_threshold)
plt.imshow(binary_b, cmap = "grey")
plt.show()
plt.imshow(binary_c, cmap = "grey")
plt.show()

# Deleting salt and pepper

def salt_and_pepper(img):
    return_img = img
    bordered_shape = [x + 2 for x in img.shape]
    bordered_img = np.full(bordered_shape, 0)
    bordered_img[1:1 + img.shape[0], 1:1 + img.shape[1]] = img
    salt = np.array([[256,256,256],[256,0,256],[256,256,256]])
    pepper = np.array([[0, 0, 0], [0, 256, 0], [0, 0, 0]])
    for x in range(len(img)):
        for y in range(len(img[0])):
            if(bordered_img[x:x+3,y:y+3]== salt).all():
                return_img[x,y] =256
            if(bordered_img[x:x+3,y:y+3] == pepper).all():
                return_img[x,y] = 0
    return return_img

clear_b = salt_and_pepper(binary_b)
clear_c = salt_and_pepper(binary_c)
plt.imshow(clear_b, cmap = "grey")
plt.show()
plt.imshow(clear_c, cmap = "grey")
plt.show()

# Grouping
def get_groups(img):
    groups = []
    for x in range(img.shape()[0]):
        for y in range(img.shape()[1]):
            is_in_group = False
            for group in groups:
                if (x,y) in group:
                    is_in_group = True
            if(is_in_group):
                continue
            else:
                groups.append(get_group(img,(x,y)))


def get_group(img,coords):
    group = []
    if img[coords[0], coords[1]] == 256:
        group.append(coords)
    coords_to_check = [(coords[0]-1,coords[1]),(coords[0]+1,coords[1]),(coords[0],coords[1]-1),(coords[0],coords[1]+1)]
    for i in coords_to_check:
        if(img[i[0],i[1]]!=0):
            group+=get_group(img,(i[0],i[1]))
    return group

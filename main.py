import math
import numpy
import numpy as np
import PIL
from matplotlib import pyplot as plt

print("Please input the step for histogram method:")
hist_step = int(input())

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

# Binarization method
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
                binary_img[x,y]=255
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
    salt = np.array([[255,255,255],[255,0,255],[255,255,255]])
    pepper = np.array([[0, 0, 0], [0, 255, 0], [0, 0, 0]])
    for x in range(len(img)):
        for y in range(len(img[0])):
            if(bordered_img[x:x+3,y:y+3]== salt).all():
                return_img[x,y] =255
            if(bordered_img[x:x+3,y:y+3] == pepper).all():
                return_img[x,y] = 0
    return return_img

clear_b = salt_and_pepper(binary_b)
clear_c = salt_and_pepper(binary_c)
plt.imshow(clear_b, cmap = "grey")
plt.show()
plt.imshow(clear_c, cmap = "grey")
plt.show()

def get_groups(binary_image):
    assert set(np.unique(binary_image)).issubset({0, 255}), "The input image must be binary (0 and 255 only)."

    visited = np.zeros_like(binary_image, dtype=bool)
    white_pixel_coords = np.argwhere(binary_image == 255)

    groups = []

    neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def flood_fill(start_coord):
        group = []
        stack = [start_coord]
        while stack:
            x, y = stack.pop()
            if visited[x, y]:
                continue
            visited[x, y] = True
            group.append((x, y))
            for dx, dy in neighbor_offsets:
                nx, ny = x + dx, y + dy
                if 0 <= nx < binary_image.shape[0] and 0 <= ny < binary_image.shape[1]:
                    if binary_image[nx, ny] == 255 and not visited[nx, ny]:
                        stack.append((nx, ny))
        return group

    for coord in white_pixel_coords:
        x, y = coord
        if not visited[x, y]:
            group = flood_fill((x, y))
            groups.append(group)

    return groups
groups_b = get_groups(binary_b)
groups_img_b = np.full((b.shape[0],b.shape[1],3),0, dtype=np.int32)
groups_c = get_groups(binary_c)
groups_img_c = np.full(c.shape,0, dtype=np.int32)

for segment in groups_b:
    color = list(np.random.choice(range(256), size=3))
    for i in color:
        i = int(i)
    for coordinates in segment:
        groups_img_b[coordinates[0],coordinates[1]] = color
for segment in groups_c:
    color = list(np.random.choice(range(256), size=3))
    for coordinates in segment:
        groups_img_c[coordinates[0],coordinates[1]] = color

plt.imshow(groups_img_b)
plt.show()
plt.imshow(groups_img_c)
plt.show()

# Histogram method
def get_local_minima_in_hist(img, step):
    histogram = np.histogram(img, bins=range(0, 256))
    thresholds = [0]
    for i in range(0,255):
        start = 0
        if i-step>0:
            start = i-step
        end = 255
        if i+step<255:
            end = i+step
        local_min = float('inf')
        min_id = 256
        for j in range (start,end):
            if(histogram[0][j]<local_min):
                local_min = histogram[0][j]
                min_id = j
        if(i==min_id):
            thresholds.append(i)
    return thresholds


b_thresholds = get_local_minima_in_hist(grey_b,hist_step)
c_thresholds = get_local_minima_in_hist(grey_c,hist_step)

def get_groups_by_hist(img, thresholds):
    groups = []
    for threshold in thresholds:
        groups.append([])
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            for i in range(len(thresholds) - 1):
                if img[x,y]>thresholds[i] and img[x,y]<=thresholds[i+1]:
                    groups[i].append((x,y))
    return groups

groups_hist_b = get_groups_by_hist(grey_b,b_thresholds)
groups_hist_img_b =np.full((b.shape[0],b.shape[1],3),0, dtype=np.int32)
for group in groups_hist_b:
    color = list(np.random.choice(range(256), size=3))
    for i in color:
        i = int(i)
    for coordinates in group:
        groups_hist_img_b[coordinates[0],coordinates[1]] = color
groups_hist_c = get_groups_by_hist(grey_c,c_thresholds)
groups_hist_img_c =np.full((c.shape[0],c.shape[1],3),0, dtype=np.int32)
for group in groups_hist_c:
    color = list(np.random.choice(range(256), size=3))
    for i in color:
        i = int(i)
    for coordinates in group:
        groups_hist_img_c[coordinates[0],coordinates[1]] = color

plt.imshow(groups_hist_img_b)
plt.show()
plt.imshow(groups_hist_img_c)
plt.show()

group_matrix_b = np.full((b.shape[0], b.shape[1]), 0, dtype=np.int32)
for i in range(len(groups_hist_b)):
    for pixels in groups_hist_b[i]:
        group_matrix_b[pixels[0],pixels[1]]=i
group_matrix_c = np.full((c.shape[0], c.shape[1]), 0, dtype=np.int32)
for i in range(len(groups_hist_c)):
    for pixels in groups_hist_c[i]:
        group_matrix_c[pixels[0],pixels[1]]=i
def get_segments(img,group_matrix):
    segments = []
    visited = np.zeros_like(img, dtype=bool)
    neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    def get_segment(start_coords, number):
        segment = []
        stack = [start_coords]
        while stack:
            x, y = stack.pop()
            if visited[x, y]:
                continue
            visited[x, y] = True
            segment.append((x, y))
            for dx, dy in neighbor_offsets:
                nx, ny = x + dx, y + dy
                if 0 <= nx < img.shape[0] and 0 <= ny < img.shape[1]:
                    if number == group_matrix[nx,ny] and not visited[nx, ny]:
                        stack.append((nx, ny))
        return segment

    for x in range(len(group_matrix)):
        for y in range(len(group_matrix[0])):
            if not visited[x,y]:
                segment=get_segment((x,y),group_matrix[x,y])
                segments.append(segment)
    return segments

segments_b = get_segments(grey_b, group_matrix_b)
segments_img_b =np.full((b.shape[0],b.shape[1],3),0, dtype=np.int32)

for segment in segments_b:
    color = list(np.random.choice(range(256), size=3))
    for i in color:
        i = int(i)
    for coordinates in segment:
        segments_img_b[coordinates[0],coordinates[1]] = color
plt.imshow(segments_img_b)
plt.show()
segments_c = get_segments(grey_c, group_matrix_c)
segments_img_c =np.full((c.shape[0],c.shape[1],3),0, dtype=np.int32)

for segment in segments_c:
    color = list(np.random.choice(range(256), size=3))
    for i in color:
        i = int(i)
    for coordinates in segment:
        segments_img_c[coordinates[0],coordinates[1]] = color
plt.imshow(segments_img_c)
plt.show()
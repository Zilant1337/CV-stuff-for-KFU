import math
import numpy
import numpy as np
import PIL
from matplotlib import pyplot as plt

ntsc_values = [0.299, 0.587, 0.114]
print("Please input the size of the kernel:")
kernel_size_input = int(input())
print("Please input the gaussian standart deviation:")
gaussian_standart_deviation = float(input())
gaussian_kernel_size = kernel_size_input - (1 - kernel_size_input % 2)
a = np.array(0)

# Creating kernel
kernel = np.zeros((gaussian_kernel_size,gaussian_kernel_size))
for x in range(0, gaussian_kernel_size):
    for y in range(0, gaussian_kernel_size):
        x_from_center = gaussian_kernel_size//2+1-x
        y_from_center = gaussian_kernel_size // 2 + 1 - y
        kernel[x,y]=1/(2*math.pi*gaussian_standart_deviation**2)*math.exp(-1*(x_from_center**2+y_from_center**2)/(2*gaussian_standart_deviation**2))


# Importing image into a list
with PIL.Image.open("lanes.jpg") as img:
    a = np.array(img)
plt.imshow(a)
plt.show()

# Greyscaling the image
grey_a = np.full((a.shape[0],a.shape[1]),0, dtype=np.float64)
for i in range (3):
    grey_a+= np.multiply(a[:,:,i],ntsc_values[i])

plt.imshow(grey_a,cmap='gray')
plt.show()

# Gaussian filtering

bordered_shape = [x + gaussian_kernel_size * 2 for x in grey_a.shape]
bordered_cummulative = np.full(bordered_shape, 0)
bordered_cummulative[gaussian_kernel_size:gaussian_kernel_size + grey_a.shape[0], gaussian_kernel_size:gaussian_kernel_size + grey_a.shape[1]] = grey_a

gaussian_blur_cummulative = np.full(grey_a.shape, 0)
for i in range(gaussian_kernel_size, bordered_cummulative.shape[0] - gaussian_kernel_size):
    for j in range(gaussian_kernel_size, bordered_cummulative.shape[1] - gaussian_kernel_size):
        slice_for_kernel = bordered_cummulative[i - gaussian_kernel_size // 2:i + gaussian_kernel_size // 2 + 1, j - gaussian_kernel_size // 2:j + gaussian_kernel_size // 2 + 1]
        slice_for_kernel = np.multiply(slice_for_kernel,kernel)
        # print(f"X: {i-gaussian_kernel_size}, Y: {j-gaussian_kernel_size}")
        gaussian_blur_cummulative[i - gaussian_kernel_size, j - gaussian_kernel_size] = np.sum(slice_for_kernel)


plt.imshow(gaussian_blur_cummulative, cmap='gray')
plt.show()

# Calculating gradient

x_kernel = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
# x_kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
y_kernel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
# y_kernel = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
gradient_magnitude = np.zeros(grey_a.shape)
gradient_x = np.zeros(grey_a.shape)
gradient_y = np.zeros(grey_a.shape)
gradient_directions = np.zeros(grey_a.shape)
rounded_gradient_directions_radians = np.zeros(grey_a.shape)
nms_grey_a = np.zeros(grey_a.shape)

bordered_shape = [x + 2 for x in grey_a.shape]
bordered_cummulative = np.full(bordered_shape, 0)
bordered_cummulative[1:1 + grey_a.shape[0], 1:1 + grey_a.shape[1]] = grey_a

for i in range(1, bordered_cummulative.shape[0] - 1):
    for j in range(1, bordered_cummulative.shape[1] - 1):
        slice_for_kernel = bordered_cummulative[i - 1:i + 2, j - 1:j + 2]
        x_gradient = np.sum(np.multiply(x_kernel,slice_for_kernel))
        y_gradient = np.sum(np.multiply(y_kernel,slice_for_kernel))

        gradient_x[i-1,j-1] = x_gradient
        gradient_y[i-1,j-1] = y_gradient
        gradient_magnitude[i-1,j-1] = math.sqrt(x_gradient**2+y_gradient**2)
        gradient_directions[i-1,j-1] = math.atan2(y_gradient,x_gradient)
        rounded_gradient_directions_radians[i - 1, j - 1] = round(gradient_directions[i - 1, j - 1] / math.radians(45)) * math.radians(45)
# Non-maximum supression

bordered_shape = [x + 2 for x in gradient_magnitude.shape]
bordered_gradient_magnitude = np.full(bordered_shape,0)
bordered_gradient_magnitude[1:1 + gradient_magnitude.shape[0], 1:1 + gradient_magnitude.shape[1]] = gradient_magnitude

for i in range(1,bordered_gradient_magnitude.shape[0]-1):
    for j in range(1,bordered_gradient_magnitude.shape[1]-1):
        if (math.degrees(rounded_gradient_directions_radians[i-1,j-1]) == 0 or math.degrees(
                rounded_gradient_directions_radians[i-1,j-1]) == 0 + 180):
            if (bordered_gradient_magnitude[i, j] > max(bordered_gradient_magnitude[i, j + 1], bordered_gradient_magnitude[i, j - 1])):
                nms_grey_a[i-1,j-1] = bordered_gradient_magnitude[i, j]
            else:
                nms_grey_a[i-1,j-1] = 0
        if(math.degrees(rounded_gradient_directions_radians[i-1,j-1])==45 or math.degrees(rounded_gradient_directions_radians[i-1,j-1])==45+180):
            if(bordered_gradient_magnitude[i,j]>max(bordered_gradient_magnitude[i-1,j+1],bordered_gradient_magnitude[i+1,j-1])):
                nms_grey_a[i-1,j-1] = bordered_gradient_magnitude[i,j]
            else:
                nms_grey_a[i-1,j-1] = 0
        if (math.degrees(rounded_gradient_directions_radians[i-1,j-1]) == 90 or math.degrees(
                rounded_gradient_directions_radians[i-1,j-1]) == 90 + 180):
            if (bordered_gradient_magnitude[i, j] > max(bordered_gradient_magnitude[i - 1, j], bordered_gradient_magnitude[i + 1, j])):
                nms_grey_a[i-1,j-1] = bordered_gradient_magnitude[i, j]
            else:
                nms_grey_a[i-1,j-1] = 0
        if (math.degrees(rounded_gradient_directions_radians[i-1,j-1]) == 135 or math.degrees(
                rounded_gradient_directions_radians[i-1,j-1]) == 135 + 180):
            if (bordered_gradient_magnitude[i, j] > max(bordered_gradient_magnitude[i - 1, j - 1], bordered_gradient_magnitude[i + 1, j + 1])):
                nms_grey_a[i-1,j-1] = bordered_gradient_magnitude[i, j]
            else:
                nms_grey_a[i-1,j-1] = 0

t_high = 200
t_low = 100

def border_specification(gradient, t_high, t_low):
    visited_matrix = np.full(grey_a.shape, -1)
    less_than_low = np.argwhere(gradient<t_low)
    higher_than_high = np.argwhere(gradient>t_high)
    neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1),(-1,-1),(-1,1),(1,-1),(1,1)]
    for coords in less_than_low:
        x,y = coords
        visited_matrix[x,y] = 0

    def recursive_neighbour_check(start_coord):
        stack = [start_coord]
        while stack:
            x, y = stack.pop()
            if visited_matrix[x, y]!=-1:
                continue
            visited_matrix[x, y] = 255
            for dx, dy in neighbor_offsets:
                nx, ny = x + dx, y + dy
                if 0 <= nx < gradient.shape[0] and 0 <= ny < gradient.shape[1]:
                    if gradient[nx, ny] > t_low and visited_matrix[nx,ny]==-1:
                        stack.append((nx, ny))
    for coords in higher_than_high:
        if(visited_matrix[coords[0],coords[1]]==-1):
            recursive_neighbour_check(coords)
    remaining = np.argwhere(visited_matrix == -1)
    for coords in remaining:
        visited_matrix[coords[0],coords[1]] = 0
    return visited_matrix

specified_borders = border_specification(nms_grey_a,t_high,t_low)

def hough_transform(img):
    diagonal = int(math.sqrt(img.shape[0]**2+img.shape[1]**2))
    rhos = np.linspace(0, diagonal, diagonal)
    thetas = np.deg2rad(np.arange(-90,180,1))

    cummulative = np.zeros((len(rhos), len(thetas)), dtype=np.float32)

    for x in range(img.shape[0]):
        for y in range (img.shape[1]):
            if img[x,y]==255:
                for id, theta in enumerate(thetas):
                    rho = int(y*np.cos(theta)+ x* np.sin(theta))
                    cummulative[rho,id] +=1

    return cummulative, rhos,thetas

cummulative,rhos,thetas = hough_transform(specified_borders)

print("Please input the size of the kernel for cummulative:")
kernel_size_input = int(input())
print("Please input the gaussian standart deviation:")
gaussian_standart_deviation = float(input())
print("Please input the threshold for non-maximum suppression:")
nms_threshold = float(input())
gaussian_kernel_size = kernel_size_input - (1 - kernel_size_input % 2)

# Creating kernel
kernel = np.zeros((gaussian_kernel_size,gaussian_kernel_size))
for x in range(0, gaussian_kernel_size):
    for y in range(0, gaussian_kernel_size):
        x_from_center = gaussian_kernel_size//2+1-x
        y_from_center = gaussian_kernel_size // 2 + 1 - y
        kernel[x,y]=1/(2*math.pi*gaussian_standart_deviation**2)*math.exp(-1*(x_from_center**2+y_from_center**2)/(2*gaussian_standart_deviation**2))

# Gaussian filtering

bordered_shape = [x + gaussian_kernel_size * 2 for x in cummulative.shape]
bordered_cummulative = np.full(bordered_shape, 0)
bordered_cummulative[gaussian_kernel_size:gaussian_kernel_size + cummulative.shape[0], gaussian_kernel_size:gaussian_kernel_size + cummulative.shape[1]] = cummulative

gaussian_blur_cummulative = np.full(cummulative.shape, 0)
for i in range(gaussian_kernel_size, bordered_cummulative.shape[0] - gaussian_kernel_size):
    for j in range(gaussian_kernel_size, bordered_cummulative.shape[1] - gaussian_kernel_size):
        slice_for_kernel = bordered_cummulative[i - gaussian_kernel_size // 2:i + gaussian_kernel_size // 2 + 1, j - gaussian_kernel_size // 2:j + gaussian_kernel_size // 2 + 1]
        slice_for_kernel = np.multiply(slice_for_kernel,kernel)
        # print(f"X: {i-gaussian_kernel_size}, Y: {j-gaussian_kernel_size}")
        gaussian_blur_cummulative[i - gaussian_kernel_size, j - gaussian_kernel_size] = np.sum(slice_for_kernel)

def nms(cummulative, threshold):
    max_value = np.max(cummulative)
    significant_value = int(threshold * max_value)
    cummulative_height, cummulative_width = cummulative.shape

    local_maximum = []
    for r in range(cummulative_height):
        for t in range(cummulative_width):
            if cummulative[r, t] > significant_value:
                neighborhood = cummulative[max(0, r - 1):min(cummulative.shape[0], r + 2), max(0, t - 1):min(cummulative.shape[1], t + 2)]
                if cummulative[r, t] == np.max(neighborhood):
                    local_maximum.append((r, t, cummulative[r, t]))

    print(f"Found {len(local_maximum)} lines with threshold {threshold * 100}% from maximum value")

    return local_maximum

nms_blurred_cummulative = nms(gaussian_blur_cummulative,nms_threshold)

def draw_lines(img, rhos, thetas, nms_cummulative, accuracy=0.8):
    height, width, _ = img.shape
    for r, t, value in nms_cummulative:
        theta = thetas[t]
        rho = rhos[r]

        for y in range(height):
            for x in range(width):
                if abs(x * np.cos(theta) + y * np.sin(theta) - rho) < accuracy:
                    img[y, x] = [0, 0, 255]

    return img

a_with_lines = draw_lines(a,rhos,thetas,nms_blurred_cummulative)

fig,axis = plt.subplots(1,3)
axis[0].imshow(specified_borders,cmap='gray')
axis[0].set_title("Specified borders (Canny)")
axis[1].imshow(gaussian_blur_cummulative, cmap='gray')
axis[1].set_title("Cummulative array")
axis[2].imshow(a_with_lines)
axis[2].set_title("Image with found lines")

plt.show()
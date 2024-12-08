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
with PIL.Image.open("road2.jpg") as img:
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
plt.imshow(specified_borders, cmap= "grey")
plt.show()
def apply_hough_transform(edges):
    height, width = edges.shape
    diagonal = int(np.sqrt(height ** 2 + width ** 2))
    thetas = np.deg2rad(np.arange(-90, 180, 1))
    rhos = np.linspace(0, diagonal, diagonal)

    phase = np.zeros((len(rhos), len(thetas)), dtype=np.float32)

    for i in range(height):
        for j in range(width):
            if edges[i, j] > 0:
                for index, theta in enumerate(thetas):
                    rho = int(j * np.cos(theta) + i * np.sin(theta))
                    phase[rho, index] += 1

    return phase, thetas, rhos


hough_transform_matrix, thetas, rhos = apply_hough_transform(specified_borders)

plt.imshow(hough_transform_matrix,cmap = "grey")
plt.show()

def suppress_nonmaximum(phase, threshold):
    max_value = np.max(phase)
    significant_value = int(threshold * max_value)
    phase_height, phase_width = phase.shape

    local_maximum = []
    for r in range(phase_height):
        for t in range(phase_width):
            if phase[r, t] > significant_value:
                neighborhood = phase[max(0, r - 1):min(phase.shape[0], r + 2), max(0, t - 1):min(phase.shape[1], t + 2)]
                if phase[r, t] == np.max(neighborhood):
                    local_maximum.append((r, t, phase[r, t]))

    print(f"Found {len(local_maximum)} lines with threshold {threshold * 100}% from maximum value")

    return local_maximum

print("Enter threshold: \n")
threshold = float(input())

local_maximum = suppress_nonmaximum(hough_transform_matrix,threshold)

def draw_lines(image_array, thetas, rhos, local_maximum, accuracy=0.8):
    height, width , _= image_array.shape

    for r, t, value in local_maximum:
        theta = thetas[t]
        rho = rhos[r]

        for y in range(height):
            for x in range(width):
                if abs(x * np.cos(theta) + y * np.sin(theta) - rho) < accuracy:
                    image_array[y, x] = [0, 0, 255]

    return image_array

plt.imshow(draw_lines(a,thetas,rhos,local_maximum),cmap = "grey")
plt.show()

import numpy as np

global IM_SIZE
IM_SIZE = 32


def gradient(image):
    gradient_hor = np.zeros(image.shape)
    gradient_vert = np.zeros(image.shape)

    if len(image.shape) == 2:  # 2D
        # Horizontal gradient ([-1 0 1] filter)
        gradient_vert[1:-1, :] = image[2:, :] - image[:-2, :]
        gradient_vert[1, :] = image[1, :] - image[0, :]  # Replicate edges

        # Horizontal gradient ([-1 0 1] filter)
        gradient_hor[:, 1:-1] = image[:, 2:] - image[:, :-2]
        gradient_hor[:, 1] = image[:, 1] - image[:, 0]  # Replicate edges
    else:  # 3D
        # Horizontal gradient ([-1 0 1] filter)
        gradient_vert[1:-1, :, :] = image[2:, :, :] - image[:-2, :, :]
        gradient_vert[1, :, :] = image[1, :, :] - image[0, :, :]  # Replicate edges

        # Horizontal gradient ([-1 0 1] filter)
        gradient_hor[:, 1:-1, :] = image[:, 2:, :] - image[:, :-2, :]
        gradient_hor[:, 1, :] = image[:, 1, :] - image[:, 0, :]  # Replicate edges

    return gradient_hor, gradient_vert


def hog(image, num_angle_bins=9, cell_size=8, block_size=2):
    grad_hor, grad_vert = gradient(image)
    grad_magnitude = np.sqrt(np.power(grad_vert, 2) + np.power(grad_hor, 2))

    if len(image.shape) > 2:
        # Keep biggest gradient magnitude amongst the color chanels
        color_max_magnitude = np.argmax(grad_magnitude, axis=2)
        grad_magnitude = np.amax(grad_magnitude, axis=2)
        indices = np.indices(image.shape[:-1])
        grad_hor = grad_hor[indices[0], indices[1], color_max_magnitude]
        grad_vert = grad_vert[indices[0], indices[1], color_max_magnitude]

    # Unsigned orientation of the gradient
    grad_orientation = np.arctan2(grad_vert, grad_hor)
    grad_orientation = grad_orientation % np.pi  # Unsigned angle

    num_cells = int(IM_SIZE / cell_size)
    histograms = np.zeros((num_angle_bins, num_cells, num_cells))

    angle_increment = np.pi / num_angle_bins
    for i in range(num_angle_bins):
        angles_inf = grad_orientation <= (angle_increment * (i + 0.5))
        angles_sup = grad_orientation > ((angle_increment * (i - 0.5)) % np.pi)
        angles = angles_inf * angles_sup
        for j in range(num_cells):
            for k in range(num_cells):
                cell_mask = np.zeros(angles.shape)
                cell_mask[j*cell_size: (j+1)*cell_size,
                          k*cell_size: (k+1)*cell_size] = 1
                histograms[i, j, k] = np.sum(angles * cell_mask * grad_magnitude)

    num_blocks = int(num_cells - block_size + 1)
    HOG_vector = np.zeros((num_angle_bins * block_size ** 2, num_blocks, num_blocks))
    for j in range(num_blocks):
        for k in range(num_blocks):
            block_hist = histograms[:, j: j+block_size, k: k+block_size]
            block_hist = block_hist.flatten() / np.linalg.norm(block_hist)
            HOG_vector[:, j, k] = block_hist

    return HOG_vector.flatten()

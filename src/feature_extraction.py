from PIL import Image
import numpy as np
import os
from scipy import stats
import pickle



def gradients_without_padding(image):

    [height, width] = image.shape[0:2]

    x_gradient = np.zeros((height, width), dtype='int')
    y_gradient = np.zeros((height, width), dtype='int')

    for i in range(0, height):
        for j in range(0, width):

            # left pixel
            if j == 0:
                left = 0
            else:
                left = image[i][j - 1][0]

            # right pixel
            if j == width - 1:
                right = 0
            else:
                right = image[i][j + 1][0]

            # upper pixel
            if i == 0:
                upper = 0
            else:
                upper = image[i - 1][j][0]

            # under pixel
            if i == height - 1:
                under = 0
            else:
                under = image[i + 1][j][0]

            # X gradient:
            x_gradient[i][j] = right - left

            # Y gradient:
            y_gradient[i][j] = under - upper


    # Now that we have calculated the X and Y gradients, we compute the final gradient magnitude representation
    x_squared = np.multiply(x_gradient.astype('int'), x_gradient.astype('int'))
    y_squared = np.multiply(y_gradient.astype('int'), y_gradient.astype('int'))

    gradient_magnitude = np.sqrt(x_squared + y_squared).astype('uint8')

    # The orientation matrix is computed as atan2(y/x)
    # NOTE : because we want the orientation to be [0, 180], negative angles have to be handled

    gradient_orientation = 180/np.pi * np.arctan2(y_gradient, x_gradient)
    gradient_orientation += np.less(gradient_orientation, 0) * 180

    return [gradient_magnitude, gradient_orientation]




def compute_one_histogram(magnitude, orientation, start, cell_shape, number_of_orientations=8 ):
    """
        Computes a histogram for one cell.
        That cell is determined by its starting index and its shape.

        Returns:
            - statistics - sum of each bin in the histogram

    """

    [start_i, start_j] = start
    [height, width] = cell_shape

    cell_mag = np.zeros(width * height, dtype='uint8')
    cell_orient = np.zeros(width * height, dtype='int')

    # iter will be the full pixel counter
    iter = 0

    # Concatenation of cell magnitudes and orientations onto their respective vectors
    for i in range(start_i, start_i + height):
        for j in range(start_j, start_j + width):

            cell_mag[iter] = magnitude[i][j]
            cell_orient[iter] = orientation[i][j]
            iter += 1

    # Once we have our magnitude and orientation vectors, we can calculate the histograms.
    # Since the numpy histogram function only returns the number of appearences in a single bin,
    # we have to use this scikit learn functions which sums the appearences inside of a bin.
    statistic, bin_edges, bin_number = stats.binned_statistic(cell_orient, cell_mag,
                                                              statistic='sum', bins = number_of_orientations)

    return statistic




def compute_histograms(magnitude, orientation, cell_shape, number_of_orientations = 8):
    """
        Inputs:
            - magnitude - ...
            - orientation - ...
            - cell_shape - (width, height) variable denoting the number of pixels in a cell

        Returns:
            - all_histograms - list of histograms

        This function iterates through all cells (of size cell_shape), without overlapping, and computes
        one histogram per cell.

        !!! The size of the final feature vector is controlled by the cell size, thus the number of cells, as well.
    """

    # Check if the cell_shape is consistent with the image shape
    [cell_height, cell_width] = np.asarray(cell_shape)
    [image_height,image_width] = magnitude.shape

    if not np.mod(image_width, cell_width) == 0:
        print("Invalid input: cell width !!!")
        return

    if not np.mod(image_height, cell_height) == 0:
        print("Invalid input: cell height !!!")
        return

    # All histograms are going to be stored in a list, so it will be easier to reference them afterwards
    all_histograms = []

    # Iterate through the image, cell by cell, and compute a histogram for each of them
    # In every iteration, (i, j) is the starting index of the current cell
    i = 0
    while i < image_height:
        j = 0
        while j < image_width:

            # if j == 1: return

            current_histogram = compute_one_histogram(magnitude, orientation, [i, j], [cell_height, cell_width])
            all_histograms.append(current_histogram)

            j += cell_width

        i += cell_height

    return all_histograms




def block_normalization(histograms, width, height, block_shape):
    """
    """

    return




def one_image_HoG_visualization(histograms, height, width, filepath):
    """
        Visualization of HoG features of a single image.
        Depending on a cell's maximum magnitude and the corresponding orientation, certain 3 pixels
        (of value 0-255) are connected, in order to represent the gradient of that particular cell.

    """

    # We use 3x3 pixels to represent each cell
    visualized = np.zeros((3*height, 3*width), dtype='uint8')

    hist_index = 0
    i = 0

    while i < 3*height:
        j = 0
        while j < 3*width:

            # Depending on the orientation and magnitude of the 'strongest' gradient in the
            # current cell, a 'line' is drawn into the corresponding 3x3 pixel area
            histogram = histograms[hist_index]
            max_mag_ind = np.argmax(histogram)
            max = histogram[max_mag_ind]

            visualized[i + 1][j + 1] = max

            if max_mag_ind == 0 or max_mag_ind == 4:
                visualized[i + 1][j]     = max
                visualized[i + 1][j + 2] = max

            elif max_mag_ind == 1 or max_mag_ind == 5:
                visualized[i][j + 2]     = max
                visualized[i + 2][j]     = max

            elif max_mag_ind == 2 or max_mag_ind == 6:
                visualized[i][j + 1]     = max
                visualized[i + 2][j + 1] = max

            elif max_mag_ind == 3 or max_mag_ind == 7:
                visualized[i][j]         = max
                visualized[i + 2][j + 2] = max

            hist_index += 1
            j += 3

        i += 3

    current_image = Image.fromarray(visualized)
    current_image.save(filepath)




def extract_HoG_features(range, cell_shape, path_to_folder, dest_path, features_path, save_images=(0, 0), save_features=1):
    """
        Inputs:
            - range - including both ends !!!

    """

    [start, end] = np.asarray(range)
    [cell_height, cell_width] = np.asarray(cell_shape)
    [save_start, save_end] = np.asarray(save_images)

    # To make sure there is no relative path shenanigans, everything here is done using absolute paths
    my_path = os.path.abspath(os.path.dirname(__file__))
    absolute_path = os.path.join(my_path, path_to_folder)
    absolute_dest_path = os.path.join(my_path, dest_path)

    # The final feature vector
    all_HoG_features = -1


    i = start
    while i <= end:

        print(i)

        current_name = str(i) + ".jpg"
        full_path = os.path.join(absolute_path, current_name)

        current_dest_name = str(i) + ".png"
        full_dest_path = os.path.join(absolute_dest_path, current_dest_name)

        # Open the current image and convert it to GRAY-SCALE
        image = Image.open(full_path).convert('LA')
        image_matrix = np.asarray(image)

        [height, width, dummy] = np.asarray(image_matrix.shape)

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Firstly, we NORMALIZE the image
        image = np.sqrt(image_matrix)

        # Calculate the gradients
        [magnitude, orientation] = gradients_without_padding(image)

        # Calculate all histograms for the current cell
        histograms = compute_histograms(magnitude, orientation, (cell_height, cell_width))

        # !!!!!!!!!!!!!!!!!!!!!!!!!
        # BASIC NORMALIZATION
        # This needs to be improved to PER-BLOCK NORMALIZATION, as opposed to whole image normalization
        histograms_normalized_array = (np.asarray(histograms) / np.max(histograms) * 255).astype('uint8')
        # histograms_normalized_array = block_normalization()
        histograms_normalized = histograms_normalized_array.tolist()

        # Reshape this NxM list into an N*M array - 1 array represents one image
        [arr_height, arr_width] = np.asarray(histograms_normalized_array.shape)
        histograms_normalized_array = np.reshape(histograms_normalized_array, arr_height * arr_width)

        # Add this array as a new row of the final feature vector
        if np.sum(all_HoG_features) == -1:
            all_HoG_features = np.array([histograms_normalized_array])
        else:
            all_HoG_features = np.append(all_HoG_features, [histograms_normalized_array], axis = 0)


        if i >= save_start and i <= save_end:
            one_image_HoG_visualization(histograms_normalized, int(height / cell_height), int(width / cell_width), full_dest_path)

        i += 1

    if save_features == 1:
        filepath = os.path.join(my_path, features_path)
        current_id = "HoG_features_" + str(start) + "_" + str(end) + ".obj"
        filepath = os.path.join(filepath, current_id)
        filehandler = open(filepath, 'wb')

        pickle.dump(all_HoG_features, filehandler)

    # logger.close()

if __name__ == '__main__':

    # extract_HoG_features(range=(0, 9000), cell_shape=(10, 10),
    #                      path_to_folder = "../dataset/data/cropped/",
    #                      dest_path = "../dataset/data/HoG/",
    #                      save_images=(1000, 1010), save_features=1)

    extract_HoG_features(range=(4000, 6000), cell_shape=(8, 8),
                         path_to_folder = "../dataset/data/cropped/bad/FSRS",
                         dest_path = "../dataset/data/HoG/",
                         save_images=(1, 10), save_features=1,
                         features_path = "../dataset/data/features/bad/FSRS")



    # SPAJANJE NOVIH OBELEZJA
    # my_path = os.path.abspath(os.path.dirname(__file__))

    # features_path = "../dataset/data/features/bad/FR/HoG_features_1_4000.obj"
    # filepath = os.path.join(my_path, features_path)

    # file = open(filepath, 'rb')
    # vector_1 = pickle.load(file)
    # file.close()

    # features_path = "../dataset/data/features/bad/FR/HoG_features_4000_6000.obj"
    # filepath = os.path.join(my_path, features_path)

    # file = open(filepath, 'rb')
    # vector_2 = pickle.load(file)
    # vector_2 = vector_2[1 : , :]
    # file.close()

    # both = np.append(vector_1, vector_2, axis = 0)

    # filepath = "../dataset/data/features/bad/FR/HoG_features_1_6000.obj"
    # filehandler = open(filepath, 'wb')
    # pickle.dump(both, filehandler)
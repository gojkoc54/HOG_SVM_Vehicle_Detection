from PIL import Image
import numpy as np
import os
import pickle
import feature_extraction as fe
import baseline_model as bm




def classify_window(window):

    predictions = []
    decisions = []

    my_path = os.path.abspath(os.path.dirname(__file__))
    path_to_folder = os.path.join(my_path, "../models")

    window_image = Image.fromarray(window)

    sizes = [(256, 256), (256, 128), (256, 128)]
    model_names = ["FR_model_1.obj", "FSRS_model_1.obj", "S_model.obj"]

    for i in range(0, 3):
        size = sizes[i]
        model_name = model_names[i]

        full_path = os.path.join(path_to_folder, model_name)

        window_resized = window_image.resize(size, Image.ANTIALIAS)

        image = np.sqrt(window_resized)

        # Calculate the gradients
        [magnitude, orientation] = fe.gradients_without_padding(image)

        # Calculate all histograms for the current cell
        histograms = fe.compute_histograms(magnitude, orientation, (8, 8))

        # BASIC NORMALIZATION
        histograms_normalized_array = (np.asarray(histograms) / np.max(histograms) * 255).astype('uint8')
        histograms_normalized = histograms_normalized_array.tolist()

        # Reshape this NxM list into an N*M array - 1 array represents one image
        [arr_height, arr_width] = np.asarray(histograms_normalized_array.shape)
        histograms_normalized_array = np.reshape(histograms_normalized_array, arr_height * arr_width)

        # LOAD MODEL
        file = open(full_path, 'rb')
        model = pickle.load(file)
        file.close()

        # MODEL PREDICTION
        prediction = model.predict(np.reshape(histograms_normalized_array, (1, -1)))
        decision = model.decision_function(np.reshape(histograms_normalized_array, (1, -1)))

        predictions.append(prediction)
        decisions.append(decision)

    if sum(predictions) > 0:
        return [1, max(decisions)]
    else:
        return [0, 0]




def draw_bounding_box(image, start, shape):

    [i, j] = start 
    [height, width] = shape 

    image[i, j : j + width] = 255
    image[i + height, j : j + width] = 255
    image[i : i + height, j] = 255
    image[i : i + height, j + width] = 255

    image[i - 1, j - 1 : j + width + 1] = 255
    image[i + height + 1, j - 1 : j + width + 1] = 255
    image[i - 1 : i + height + 1, j - 1] = 255
    image[i - 1 : i + height + 1, j + width + 1] = 255

    return image 




def find_cars(image_name, image_shape, window_sizes, found_name, folder_path, h_lim, w_lim, thresh):

    my_path = os.path.abspath(os.path.dirname(__file__))
    path_to_img_folder = os.path.join(my_path, folder_path)
    full_img_path = os.path.join(path_to_img_folder, image_name)

    mask = np.zeros(image_shape)

    # Open the current image and convert it to GRAY-SCALE
    # image = Image.open(full_img_path).convert('LA')
    image_rgb = np.array(Image.open(full_img_path))
    image_matrix = np.array(Image.open(full_img_path).convert('LA'))

    for window_size in window_sizes:

        [height, width, dummy] = image_matrix.shape

        [window_height, window_width] = window_size
        delta_w = int(window_width / 3)
        delta_h = int(window_height / 3)

        max_decision = []
        max_i = []
        max_j = []
        found = False 

        mask = np.zeros(image_shape)

        i = 680
        # while i + window_height < height:
        while i + window_height < h_lim:
            j = 0
            # while j + window_width < width:
            while j + window_width < w_lim:

                if mask[i][j] == 0:

                    current_window = image_matrix[i : i+window_height, j : j+window_width]
                    [prediction, decision] = classify_window(current_window)

                    # Kad prvi put nadjes, detaljnije potrazi
                    if (prediction == 1):

                        curr_mask = np.zeros(image_shape)

                        if (i - int(window_height * 0.7) > 0):
                            up = i - int(window_height * 0.7)
                        else:
                            up = 0

                        if (j - int(window_width * 0.4) > 0):
                            left = j - int(window_width * 0.4)
                        else:
                            left = 0

                        curr_mask[up : i+window_height, left : j+window_width] = 1

                        curr_max = decision
                        curr_i = i
                        curr_j = j 

                        classified = 0

                        i_new = i 
                        while (i_new + window_height < height) and (i_new < i + delta_h):
                            j_new = j
                            while (j_new + window_width < width) and (j_new < j + delta_w):

                                current_window = image_matrix[i_new : i_new+window_height, j_new : j_new+window_width]
                                [prediction, decision] = classify_window(current_window)

                                if (prediction == 1):
                                    classified += 1
                                    if (i_new - int(window_height * 0.7) > 0):
                                        up = i_new - int(window_height * 0.7)
                                    else:
                                        up = 0

                                    if (j_new - int(window_width * 0.4) > 0):
                                        left = j_new - int(window_width * 0.4)
                                    else:
                                        left = 0

                                if i_new < curr_i:
                                    curr_i = i_new
                                if j_new < curr_j:
                                    curr_j = j_new

                                if (decision > curr_max):
                                    curr_max = decision
                                    # curr_i = i_new
                                    # curr_j = j_new

                                    curr_mask[up : i_new+window_height, left : j_new+window_width] = 1      
                                
                                j_new += 20
                            
                            i_new += 20

                        thresh = 4
                        if classified > thresh:
                            max_decision.append(curr_max)
                            max_i.append(curr_i)
                            max_j.append(curr_j)

                            mask += curr_mask
                                

                j += 50

            i += 50

        for iter in range(0, len(max_decision)):
            if max_decision[iter] > 0:
                final_image = draw_bounding_box(image_rgb, [max_i[iter], max_j[iter]], [window_height, window_width])
                final_image = Image.fromarray(final_image)
                final_image.save(my_path + '/' + folder_path + '/nadjene/' + found_name)

        print(image_name + " : " + str(max_decision))

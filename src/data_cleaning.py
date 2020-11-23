from PIL import Image
import numpy as np
import os


def extract_labels(label_path):

    current_label = open(label_path, 'r')

    view = current_label.readline()
    bounding_box_num = current_label.readline()
    bounding_box = current_label.readline()

    view = int(view)
    bounding_box = np.asarray(bounding_box[:-1].split(" ")).astype('int')

    good_view = 0
    if view in [1, 2]:
        # good_view = 'FR'
        good_view = 0
    
    elif view == 3:
        # good_view = 'S'
        good_view = 0

    
    elif view in [4, 5]:
        good_view = 'FSRS'
        # good_view = 0

    start = [bounding_box[0], bounding_box[1]]
    end = [bounding_box[2], bounding_box[3]]

    return [good_view, start, end]




def extract_boounding_box(image, start, end, non_car = False):

    if non_car == True:

        mode = "shrink"
        
        if mode == "left_up":
            delta = 150
            additional_delta = 20

            if start[0] - delta > 0:
                start[0] = start[0] - delta  
            else:
                start[0] = 0
            
            if start[1] - delta > 0:
                start[1] = start[1] - delta 
            else:
                start[1] = 0

            end[0] = end[0] - delta - additional_delta
            end[1] = end[1] - delta - additional_delta

        if mode == "right_up":
            delta = 200
            additional_delta = 20

            if end[0] + delta < image.shape[1] - 1:
                end[0] = end[0] + delta  
            else:
                end[0] = image.shape[1] - 1
            
            if start[1] - delta > 0:
                start[1] = start[1] - delta 
            else:
                start[1] = 0

            start[0] = start[0] + delta + additional_delta
            end[1] = end[1] - delta - additional_delta

        if mode == "shrink":
            delta = 120
  
            start[0] = start[0] + delta  
            start[1] = start[1] + delta

            if end[0] - delta > start[0]:
                end[0] = end[0] - delta
            else:
                end[0] = start[0] + 50

            if end[1] - delta > start[1]:
                end[1] = end[1] - delta 
            else:
                end[1] = start[1] + 50 
            

    width = end[0] - start[0]
    height = end[1] - start[1]

    cropped = np.zeros((height, width), dtype='uint8')

    for i in range(start[1], end[1]):
        for j in range(start[0], end[0]):

            cropped[i - start[1]][j - start[0]] = image[i][j][0]

    return cropped




def cleaning_with_bounding_boxes(path_to_names, path_to_labels, path_to_images, new_path, i_start = [0, 0, 0, 0], resize = 1, 
                                 resize_shape_FR = None, resize_shape_S = None, resize_shape_FSRS = None):
    
    # Firstly, we open the .txt file containing paths from the 'image' folder to each of the images

    my_path = os.path.abspath(os.path.dirname(__file__))
    path_to_names = os.path.join(my_path, path_to_names)
    names_file = open(path_to_names, 'r')

    [i, count_fr, count_s, count_fsrs] = i_start
    
    while True:

        # if (i == 20): break
        if i < 4680:

            current_name = names_file.readline()
            if not current_name: break
            i += 1

            continue

        if i > 6000:
            break

        # Read one image name (i.e. path to the image from the 'image' folder)
        current_name = names_file.readline()

        if not current_name: break

        # !!! remove '\n' from the string
        current_name = current_name[ : -4]

        # Concatenate the path to the 'image' folder and the current_name path,
        # in order to get the full path to the current image
        full_image_path = os.path.join(my_path, path_to_images)
        full_image_path = full_image_path + current_name + 'jpg'

        full_label_path = os.path.join(my_path, path_to_labels)
        full_label_path = full_label_path + current_name + 'txt'

        [good_view, start, end] = extract_labels(full_label_path)

        if good_view == 0:
            continue

        elif good_view == 'S':
            resize_shape = resize_shape_S
            count_s += 1
            curr_count = count_s
        
        elif good_view == 'FR':
            resize_shape = resize_shape_FR
            count_fr += 1
            curr_count = count_fr
        
        elif good_view == 'FSRS':
            resize_shape = resize_shape_FSRS
            count_fsrs += 1
            curr_count = count_fsrs

        current_image = Image.open(full_image_path, 'r')

        cropped = extract_boounding_box(np.asarray(current_image), start, end, non_car = True)
        current_image = Image.fromarray(cropped)

        # If it was requested that way, resize the image:
        if resize == 1:
            current_image = current_image.resize(resize_shape, Image.ANTIALIAS)

        # Rename this image and save it to the new destination path
        # new_path += good_view + '/'
        destination_name = str(curr_count) + ".jpg"
        full_destination_path = os.path.join(my_path, new_path + good_view + '/')
        full_destination_path = full_destination_path + destination_name

        current_image.save(full_destination_path)

        i += 1

    names_file.close()
    return [i, count_fr, count_s, count_fsrs]

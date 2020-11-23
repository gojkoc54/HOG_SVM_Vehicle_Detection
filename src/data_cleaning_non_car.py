from PIL import Image
import numpy as np
import os

# Non-car set cleaning

def crop_one_image(image, new_path, i, size):

    # (i, j) => height, width

    # The heighest 4 crops
    # (0, 0) => 306, 407
    current_crop = image[0 : 306, 0 : 407, 0]
    current_image = Image.fromarray(current_crop)
    current_image = current_image.resize(size, Image.ANTIALIAS)

    current_path = os.path.join(new_path, str(i) + '.jpg')
    current_image.save(current_path)

    # (0, 407) => 306, 407
    current_crop = image[0 : 306, 407 : 814, 0]
    current_image = Image.fromarray(current_crop)
    current_image = current_image.resize(size, Image.ANTIALIAS)

    current_path = os.path.join(new_path, str(i + 1) + '.jpg')
    current_image.save(current_path)

    # (0, 814) => 306, 407
    current_crop = image[0 : 306, 814 : 1221, 0]
    current_image = Image.fromarray(current_crop)
    current_image = current_image.resize(size, Image.ANTIALIAS)

    current_path = os.path.join(new_path, str(i + 2) + '.jpg')
    current_image.save(current_path)

    # (0, 1221) => 306, 407
    current_crop = image[0 : 306, 1221 : 1628, 0]
    current_image = Image.fromarray(current_crop)
    current_image = current_image.resize(size, Image.ANTIALIAS)

    current_path = os.path.join(new_path, str(i + 3) + '.jpg')
    current_image.save(current_path)

    # Lower 4 crops
    # (306, 0) => 307, 407
    current_crop = image[306 : 613, 0 : 407, 0]
    current_image = Image.fromarray(current_crop)
    current_image = current_image.resize(size, Image.ANTIALIAS)

    current_path = os.path.join(new_path, str(i + 4) + '.jpg')
    current_image.save(current_path)

    # (306, 407) => 307, 407
    current_crop = image[306 : 613, 407 : 814, 0]
    current_image = Image.fromarray(current_crop)
    current_image = current_image.resize(size, Image.ANTIALIAS)

    current_path = os.path.join(new_path, str(i + 5) + '.jpg')
    current_image.save(current_path)

    # (306, 814) => 307, 407
    current_crop = image[306 : 613, 814 : 1221, 0]
    current_image = Image.fromarray(current_crop)
    current_image = current_image.resize(size, Image.ANTIALIAS)

    current_path = os.path.join(new_path, str(i + 6) + '.jpg')
    current_image.save(current_path)

    # (306, 1221) => 307, 407
    current_crop = image[306 : 613, 1221 : 1628, 0]
    current_image = Image.fromarray(current_crop)
    current_image = current_image.resize(size, Image.ANTIALIAS)

    current_path = os.path.join(new_path, str(i + 7) + '.jpg')
    current_image.save(current_path)

    # Bigger 2 crops
    # (613, 0) => 613, 814
    current_crop = image[613 : 1226, 0 : 814, 0]
    current_image = Image.fromarray(current_crop)
    current_image = current_image.resize(size, Image.ANTIALIAS)

    current_path = os.path.join(new_path, str(i + 8) + '.jpg')
    current_image.save(current_path)

    # (613, 814) => 613, 814
    current_crop = image[613 : 1226, 814 : 1628, 0]
    current_image = Image.fromarray(current_crop)
    current_image = current_image.resize(size, Image.ANTIALIAS)

    current_path = os.path.join(new_path, str(i + 9) + '.jpg')
    current_image.save(current_path)



def crop_images(path_to_images, new_path, size):

    my_path = os.path.abspath(os.path.dirname(__file__))
    path_to_images = os.path.join(my_path, path_to_images)
    path_to_dest = os.path.join(my_path, new_path)

    i = 1
    new_images = 1

    while i <= 9006:

        current_name = str(i) + ".jpg"
        current_path = os.path.join(path_to_images, current_name)

        current_image = Image.open(current_path, 'r')
        current_image_matrix = np.asarray(current_image)

        crop_one_image(current_image_matrix, path_to_dest, new_images, size)

        new_images += 10
        i += 10




def resize_images(path_to_images, new_path_to_images, new_shape, extension):

    my_path = os.path.abspath(os.path.dirname(__file__))
    path_to_images = os.path.join(my_path, path_to_images)
    new_path_to_images = os.path.join(my_path, new_path_to_images)

    i = 1

    while i <= 9006:

        current_name = str(i) + extension
        current_path = os.path.join(path_to_images, current_name)
        new_path =  os.path.join(new_path_to_images, current_name)

        current_image = Image.open(current_path, 'r')
        current_image = current_image.resize(new_shape, Image.ANTIALIAS)
        current_image.save(new_path)

        i += 1

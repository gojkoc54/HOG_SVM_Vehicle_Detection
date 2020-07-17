import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sklearn.metrics
import os
import pickle
from PIL import Image
import feature_extraction as fe



def load_features(non_car_folder, non_car_name, semi_car_folder, semi_car_name, car_folder, car_name_1 = None, car_name_2 = None):

    # Load NON-CAR features ==>> NON_CAR_VECTOR

    path_to_folder = non_car_folder
    vector_name = non_car_name

    my_path = os.path.abspath(os.path.dirname(__file__))
    path_to_folder = os.path.join(my_path, path_to_folder)
    full_path = os.path.join(path_to_folder, vector_name)

    file = open(full_path, 'rb')

    # NON-CAR VECTOR
    non_car_vector = pickle.load(file)

    file.close()


    # Load SEMI-CAR features ==>> SEMI_CAR_VECTOR

    path_to_folder = semi_car_folder
    vector_name = semi_car_name

    my_path = os.path.abspath(os.path.dirname(__file__))
    path_to_folder = os.path.join(my_path, path_to_folder)
    full_path = os.path.join(path_to_folder, vector_name)

    file = open(full_path, 'rb')

    # SEMI-CAR VECTOR
    semi_car_vector = pickle.load(file)

    file.close()

    # FINAL NON-CAR VECTOR
    non_car_vector = np.append(non_car_vector, semi_car_vector, axis = 0)

    # Load CAR features ==>> CAR_VECTOR_1 ; CAR_VECTOR_2 ==>>  CAR_VECTOR

    path_to_folder = car_folder
    vector_1_name = car_name_1

    if not car_name_2 == None:
        vector_2_name = car_name_2

    my_path = os.path.abspath(os.path.dirname(__file__))
    path_to_folder = os.path.join(my_path, path_to_folder)
    full_path_1 = os.path.join(path_to_folder, vector_1_name)
    
    if not car_name_2 == None:
        full_path_2 = os.path.join(path_to_folder, vector_2_name)

    file_1 = open(full_path_1, 'rb')

    if not car_name_2 == None:
        file_2 = open(full_path_2, 'rb')

    car_vector_1 = pickle.load(file_1)
    
    if not car_name_2 == None:
        car_vector_2 = pickle.load(file_2)

    # CAR VECTOR
    if not car_name_2 == None:
        car_vector = np.append(car_vector_1, car_vector_2, axis = 0)
    else:
        car_vector = car_vector_1

    file_1.close()

    if not car_name_2 == None:
        file_2.close()

    return [non_car_vector, car_vector]




def train_save_model(non_car_features, car_features, path_to_folder, model_name,  save = 1):

    my_path = os.path.abspath(os.path.dirname(__file__))
    path_to_folder = os.path.join(my_path, path_to_folder)
    full_path = os.path.join(path_to_folder, model_name)

    # LABELS:
    non_car_labels = np.zeros(non_car_features.shape[0])
    car_labels = np.ones(car_features.shape[0])

    # TRAIN / TEST SPLIT:
    [train_non_car_feat, test_non_car_feat] = train_test_split(non_car_features, shuffle=False, test_size = 0.2)
    [train_non_car_lab, test_non_car_lab] = train_test_split(non_car_labels, shuffle=False, test_size = 0.2)

    [train_car_feat, test_car_feat] = train_test_split(car_features, shuffle=False, test_size = 0.2)
    [train_car_lab, test_car_lab] = train_test_split(car_labels, shuffle=False, test_size = 0.2)

    train_features = np.append(train_car_feat, train_non_car_feat, axis = 0)
    train_labels = np.append(train_car_lab, train_non_car_lab, axis = 0)
    # SHUFFLE
    train_features, train_labels = shuffle(train_features, train_labels, random_state=0)

    test_features = np.append(test_car_feat, test_non_car_feat, axis = 0)
    test_labels = np.append(test_car_lab, test_non_car_lab, axis = 0)

    # MODEL TRAINING
    model = SVC(C = 1e-5, kernel = 'linear')

    model.fit(train_features, train_labels)


    # MODEL PREDICTION
    prediction = model.predict(test_features)

    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(test_labels, prediction).ravel()

    accuracy = sklearn.metrics.accuracy_score(test_labels, prediction)

    print('True positive: ', tp)
    print('False positive: ', fp)
    print('True negative: ', tn)
    print('False negative: ', fn)
    print('Accuracy: ', accuracy)

    if save == 1:

        my_path = os.path.abspath(os.path.dirname(__file__))
        path_to_folder = os.path.join(my_path, path_to_folder)
        full_path = os.path.join(path_to_folder, model_name)

        filehandler = open(full_path, 'wb')

        pickle.dump(model, filehandler)




def predict_one_image(img_name, model_name, new_shape, cell_shape, window=0):

    my_path = os.path.abspath(os.path.dirname(__file__))

    path_to_img_folder = os.path.join(my_path, "../new_images/car/jovica/cropped")
    path_to_folder = os.path.join(my_path, "../models")

    full_path = os.path.join(path_to_folder, model_name)
    full_img_path = os.path.join(path_to_img_folder, img_name)

    # Open the current image and convert it to GRAY-SCALE
    image = Image.open(full_img_path).convert('LA')
    image = image.resize(new_shape, Image.ANTIALIAS)
    image_matrix = np.asarray(image)

    [height, width, dummy] = np.asarray(image_matrix.shape)

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Firstly, we NORMALIZE the image
    image = np.sqrt(image_matrix)

    # Calculate the gradients
    [magnitude, orientation] = fe.gradients_without_padding(image)

    cell_height, cell_width = cell_shape
    # Calculate all histograms for the current cell
    histograms = fe.compute_histograms(magnitude, orientation, (cell_height, cell_width))


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
    
    print("DECISION : ", decision)
    
    if prediction == 1:
        prediction_str = "CAR"
    else:
        prediction_str = "NON-CAR"

    print("PREDICTION : ", prediction_str)

    
    return 1


if __name__ == '__main__':

    test = 0

    if test == 0:
        img_name = ""

        while not img_name == "stop":

            img_name = input("Enter the image name (or 'stop') : ")

            if img_name == 'stop': break

            print("____________________________________________________________")
            predict_one_image(img_name, "FR_model_1.obj", (256, 256), (8, 8))
            predict_one_image(img_name, "S_model.obj", (256, 128), (8, 8))
            predict_one_image(img_name, "FSRS_model_1.obj", (256, 128), (8, 8))


    if test == 1:
        [non_car_features, car_features] = load_features(
                    non_car_folder = "../belgium_dataset/features/FR",
                    non_car_name = "HoG_features_1_9006.obj",
                    semi_car_folder = "../dataset/data/features/bad/FR",
                    semi_car_name = "HoG_features_1_6000.obj",
                    car_folder = "../dataset/data/features/FR",
                    car_name_1 = "HoG_features_1_8765.obj"
                    )

        path_to_folder = "../models"
        model_name = "FR_model_2.obj"

        train_save_model(non_car_features, car_features, path_to_folder, model_name)

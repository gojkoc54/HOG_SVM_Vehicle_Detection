import numpy as np
import os
import cv2
import glob
    

def sample_video():

    my_path = os.path.abspath(os.path.dirname(__file__))
    path_to_video = os.path.join(my_path, "../new_images/car/jovica/GH010457.MP4")
    path_to_dest = os.path.join(my_path, "../new_images/car/jovica/frames/GH010457")

    cap = cv2.VideoCapture(path_to_video)


    # Preskoci 1500 frejmova pa onda kreni da citas jedan po jedan

    i = 0
    img_count = 1

    while(cap.isOpened()):
        
        ret, frame = cap.read()

        if i > 4550:
            break

        if i > 2100:
            step = 10
        else:
            step = 2100

        if i % step == 0:
            
            # crop the frame
            # frame = frame[190 : , : 1300, : ]

            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('frame',gray)
            # cv2.waitKey(0)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            
            name = str(img_count) + ".png"
            full_dest_path = os.path.join(path_to_dest, name)
            cv2.imwrite(full_dest_path, frame)
            
            img_count += 1

        i += 1
        

    cap.release()
    cv2.destroyAllWindows()




def assemble_video():

    my_path = os.path.abspath(os.path.dirname(__file__))
    # path_to_folder = os.path.join(my_path, "../new_images/car/iva/final/2/frames/cropped/nadjene")
    path_to_folder = os.path.join(my_path, "../benchmark/FOTO+VIDEO/cmrs-png/nadjene")

    img_array = []
    for i in range(90, 145):

        if i in [112, 115, 121, 124, 125]:
            continue
        
        print(i) 

        name = "image-" + str(i).zfill(4) + ".pgm"
        # filename = os.path.join(path_to_folder, str(i) + ".pgm")
        filename = os.path.join(path_to_folder, name)
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    
    destname = os.path.join(path_to_folder, "detected_video_1.avi")
    out = cv2.VideoWriter(destname,cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()




if __name__ == '__main__':

    sample_video()
import cv2
import os
from multiprocessing import Pool
import time
from preprocess import preprocess_img

class DataGen():
    def __init__(self):
        self.save_dirs = ['./dataset/singapore/', './dataset/not_singapore/'] # directory must exist before running script
        self.dir_names = ['./downloads/', './Not_Singapore_downloads/']

    # function to preprocess and save a single image
    def process_and_save(self, file):
        filename = file.split("/")[-1]
        if filename[-4:] == "json":
            return
        if os.path.isfile(file):
            training_image = preprocess_img(file)
            cv2.imwrite(self.curr_save_dir + "preprocessed_" + filename, training_image)
            return 0

    # uses multiprocessing to preprocess all images in the dataset
    def process_all(self):
        for dir, save_dir in zip(self.dir_names, self.save_dirs):
            self.curr_dir = dir
            self.curr_save_dir = save_dir
            filenames = os.listdir(dir)
            print(f"Processing {len(filenames)} in directory: ")
            print(dir + "\n")
            filenames = [dir + file for file in filenames]
            with Pool(processes=12) as pool:
                results = pool.map(self.process_and_save, filenames)

if __name__ == "__main__":
    data_generator = DataGen()
    print("Beginning preprocessing")
    start_time = time.time()
    data_generator.process_all()
    print(f"Processing complete in {time.time()-start_time} seconds")
# Singapore Bot
The goal of this project is to develop a preprocessing approach and train a neural network to classify images from Google Streetview as either from Singapore or not from Singapore.

## Usage
With the current setup, to run the preprocessing on all of the images, first move the two folders of images from the maps api into the same directory as these scripts. Then, create a directory named "dataset", and within that directory create two subdirectories, "singapore" and "not_singapore". Then, executing the gen_data script should apply preprocessing to all images and save them in these directories.

import tensorflow as tf
import os
from preprocess import preprocess_img
import cv2

saved_model = tf.keras.models.load_model("model.h5")

test_dir = "./images_for_testing/" # i put images that i just screenshotted from streetview in here

count = 0
correct = 0
for filename in os.listdir(test_dir):
    count += 1
    # assemble full path to file
    file = os.path.join(test_dir, filename)
    
    if not os.path.isfile(file):
        count -= 1
        break
    
    # determine whether test file is from singapore based on filename
    from_singapore = (filename.split(" ")[0] == '1')

    # preprocess and reshape image to prepare it to be input into model
    preprocessed_img = preprocess_img(file)
    preprocessed_img = preprocessed_img.reshape(1, 240, 240, 3)

    # use model to generate prediction
    predicted = saved_model.predict(preprocessed_img)

    # there are two outputs from the model, likelihood of it being not from singapore
    # and the likelihood it is from singapore
    # compare these values to determine what category is selected by the model
    pred_from_singapore = predicted[0][0] > .5
    confidence = (predicted[0][0] if pred_from_singapore else 1 - predicted[0][0])*100

    # display filename and results
    print(f"prediction {predicted}")
    print(f"File: {filename}")
    print(f"From Singapore: {from_singapore}")
    print(f"Predicted: {pred_from_singapore}")
    print(f"Confidence: {confidence:.2f}%")
    if pred_from_singapore == from_singapore:
        print("Prediction was correct")
        correct += 1
    else:
        print("Prediction was incorrect")
    _ = input("...enter to continue...")
    print()
print(f"Overall accuracy: {((correct/count)*100):.2f}")
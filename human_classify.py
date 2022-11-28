import PySimpleGUI as sg
from PIL import Image
import os
import io
import random
import tensorflow as tf
from preprocess import preprocess_img

LAYOUT = [[sg.Push(), sg.Text("Guess where the streetview image is from!"), sg.Push()],
            [sg.Image(key="-STREETVIEW-", data=None), sg.Column([[sg.Text("", key="-RESULT-")], [sg.Text("", key="-FILE-")], [sg.Button(button_text="Singapore", key="-BTN1-")], [sg.Button(button_text="Not Singapore", key="-BTN2-")]])],
            [sg.Push(), sg.Text(key="-LBL-"), sg.Push()]]

class HumanClassify:
    def __init__(self):
        self.saved_model = tf.keras.models.load_model("model.h5")

        self.test_dirs = ["./images_for_testing/singapore" , "./images_for_testing/not_singapore"]

        self.singapore_files = []
        self.not_singapore_files = []

        for filename in os.listdir(self.test_dirs[0]):
                # assemble full path to file
                file = self.test_dirs[0] + "/" + filename
                self.singapore_files.append(file)

        for filename in os.listdir(self.test_dirs[1]):
                # assemble full path to file
                file = self.test_dirs[1] + "/" + filename
                self.not_singapore_files.append(file)

        random.shuffle(self.singapore_files)
        random.shuffle(self.not_singapore_files)

        self.singapore_idx = 0
        self.not_singapore_idx = 0
        self.max_idx = min(len(self.singapore_files), len(self.not_singapore_files))

        self.total = 0
        self.total_correct = 0
        self.model_correct = 0

        self.guessing = True

        self.window = sg.Window('Singapore or not?', LAYOUT, finalize=True, size=(700, 570))

        # set initial image
        self.get_rand_image()
        image = Image.open(self.curr_image)
        image.thumbnail((500,500))
        bio = io.BytesIO()
        image.save(bio, format="PNG") 
        self.window["-STREETVIEW-"].update(data=bio.getvalue())

    def get_rand_image(self):
        self.singapore = bool(random.getrandbits(1))
        if self.singapore:
            image = self.singapore_files[self.singapore_idx]
            self.singapore_idx += 1
        else:
            image = self.not_singapore_files[self.not_singapore_idx]
            self.not_singapore_idx += 1
        
        self.curr_image = image

    def read(self):
        return self.window.read()

    def close(self):
        self.window.close()

    def state(self):
        return self.guessing
    
    def guess(self, guess_singapore):
        if guess_singapore and self.singapore or not guess_singapore and not self.singapore:
            self.total += 1
            self.total_correct += 1
            return True
        else:
            self.total += 1
            return False

    def show_result(self, correct):
        self.guessing = False
        
        self.window["-BTN1-"].update("Next")
        self.window["-BTN2-"].update("Quit")

        self.window["-FILE-"].update(self.curr_image.split("/")[-1])

        self.window["-LBL-"].update(self.test_model())

        if not correct:
            self.window["-RESULT-"].update("Incorrect")
        else:
            self.window["-RESULT-"].update("Correct!")
    
    def next_image(self):
        self.guessing = True

        self.get_rand_image()
        image = Image.open(self.curr_image)
        image.thumbnail((500,500))
        bio = io.BytesIO()
        image.save(bio, format="PNG") 
        self.window["-STREETVIEW-"].update(data=bio.getvalue())

        self.window["-BTN1-"].update("Singapore")
        self.window["-BTN2-"].update("Not Singapore")

        self.window["-FILE-"].update("")

        self.window["-LBL-"].update("")

    def show_results(self):
        print("Thanks for playing!")
        print(f"Your accuracy was {(self.total_correct/self.total)*100:.2f}")
        print(f"Model accuracy was {(self.model_correct/self.total)*100:.2f}")

    def test_model(self):
        # preprocess and reshape image to prepare it to be input into model
        preprocessed_img = preprocess_img(self.curr_image)
        preprocessed_img = preprocessed_img.reshape(1, 240, 240, 3)

        # use model to generate prediction
        predicted = self.saved_model.predict(preprocessed_img)

        # there are two outputs from the model, likelihood of it being not from singapore
        # and the likelihood it is from singapore
        # compare these values to determine what category is selected by the model
        pred_from_singapore = predicted[0][0] > .5
        confidence = (predicted[0][0] if pred_from_singapore else 1 - predicted[0][0])*100

        if pred_from_singapore == self.singapore:
            res = "Prediction was correct with confidence of "
            self.model_correct += 1
        else:
            res = "Prediction was incorrect with confidence of "

        res += str(round(confidence, 2)) + "%"
        return res

if __name__ == "__main__":
    human_classify = HumanClassify()
    while True:
        event, values = human_classify.read()
        if event == sg.WIN_CLOSED or event == "Exit":
                    human_classify.close()
                    break
        elif event == "-BTN1-":
            if human_classify.state():
                correct = human_classify.guess(True)
                human_classify.show_result(correct)
            else:
                human_classify.next_image()
        elif event == "-BTN2-":
            if human_classify.state():
                correct = human_classify.guess(False)
                human_classify.show_result(correct)
            else:
                human_classify.show_results()
                break

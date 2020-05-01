# TensorFlow Project
by Javier Alonso Alonso

In this project, I first develop code for an image classifier built with TensorFlow, then I convert it into a command line application.


### Data

We have 8189 flower pictures labeled as 102 different classes and we have to create the app that classify any new picture


### command line application

For running go in a cmd to the path where the predict.py is located. Run the next:

Basic usage:

$ python predict.py image.jpg saved_model_file

Options:

--top_k : Return the top KK most likely classes:
$ python predict.py image.jpg saved_model_file --top_k KK

--category_names : Path to a JSON file mapping labels to flower names:
$ python predict.py image.jpg saved_model_file --category_names map.json

the image files must be located in a folder called "test_images" and the model must be located in a folder called "models"
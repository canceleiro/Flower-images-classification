import warnings
warnings.filterwarnings('ignore')
import numpy as np
import json
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

import argparse

#Initialize parser
parser = argparse.ArgumentParser()

#Add the parameters
parser.add_argument('image_file')  
parser.add_argument('file_model')  
parser.add_argument('--label_map', default = 'label_map.json')  
parser.add_argument('--top_k', type = int, default = 6)

args = parser.parse_args()

def process_image(path):
    pic = tf.cast(path, tf.float32)
    pic = tf.image.resize(pic, (224, 224))
    pic /= 255
    pic = np.array(pic)
    return pic

def predict(image_file, file_model, top_k, label_map):
    with open(label_map, 'r') as f:
        class_names = json.load(f)
        
    model= tf.keras.models.load_model('models/'+file_model,custom_objects={'KerasLayer':hub.KerasLayer})
    
    image_path = './test_images/'+image_file
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    pic = np.expand_dims(processed_test_image, axis=0)
    
    
    ps = model.predict(pic)
    
    
    
    
    topk =ps.argsort()[0][-top_k:]
    
    probs = []
    classes = []
    j= 0
    for n in topk:
        probs.append(ps[0][n])
        classes.append(class_names[str(n+1)])
        j =  j+1
    print("hellooooooooo")
    print("The top",top_k, "type of flowers are", classes)
    print("and their probabilities to be true are",probs)
    print(classes)

    return probs, classes



predict(args.image_file, args.file_model, args.top_k, args.label_map)

import os
import json

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from model import GoogLeNet


def main():
    IMG_SIZE = (224, 224)   
    
    # load image
    img_path = "data/bee.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    # resize image to 224x224
    img = img.resize(IMG_SIZE)
    plt.imshow(img)
    
    
    preprocess_input = tf.keras.layers.Rescaling(1./127.5, offset=-1) 
    
    img = preprocess_input(np.array(img)[None,...])
    
    # read class_indict
    json_path = 'data/class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    
    with open(json_path, "r") as f:
        class_indict = json.load(f)
        
     # create model
    model = GoogLeNet(class_num=2, aux_logits=False)
    model_path = "03_googlenet/tf/myGoogLeNet"
    assert os.path.exists(model_path), "file: '{}' dose not exist.".format(model_path)
    model.load_weights(model_path)
    
    # prediction
    predict = np.squeeze(model.predict(img))
    predict_class = np.argmax(predict)
    
    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_class)],
                                                predict[predict_class])
    plt.title(print_res)
    
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                    predict[i]))
    
    plt.show()

if __name__ == "__main__":
    main()

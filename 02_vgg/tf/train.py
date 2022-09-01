import os
import json
import pathlib
import math

import tensorflow as tf
import matplotlib.pyplot as plt

from model import vgg

def main():
    BATCH_SIZE = 4
    IMG_SIZE = (224, 224)   
    
    data_root = os.path.abspath(os.getcwd())  # get data root path
    image_path = os.path.join(data_root, "data", "hymenoptera_data")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dir = os.path.join(image_path, "train")
    validation_dir = os.path.join(image_path, "val")
        
    epochs = 30
    
    train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE)

    validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                    shuffle=False,
                                                                    batch_size=BATCH_SIZE,
                                                                    image_size=IMG_SIZE)
    
    total_train = len(train_dataset.file_paths)
    class_names = train_dataset.class_names
    # transform value and key of dict
    inverse_dict = dict((str(val), key) for val, key in enumerate(class_names))
    # write dict into json file
    json_str = json.dumps(inverse_dict, indent=4)
    with open('02_vgg/tf/class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    
    total_val =  len(validation_dataset.file_paths)
    print("using {} images for training, {} images for validation.".format(total_train,
                                                                           total_val))
    
    preprocess_input = tf.keras.layers.Rescaling(1./127.5, offset=-1) 
    data_augmentation = tf.keras.Sequential([
                            tf.keras.layers.RandomFlip('horizontal'),
                            tf.keras.layers.RandomRotation(0.2),
                        ])
    
    train_dataset = train_dataset.map(lambda x, y: (preprocess_input(x), y))
    train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x), y))
    train_dataset = train_dataset.repeat()
    
    validation_dataset = validation_dataset.map(lambda x, y: (preprocess_input(x), y))
    
    model = vgg("vgg16", (*IMG_SIZE,3), num_classes=2)
    model.summary()
    
    # using keras high level api for training
    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])
    
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='02_vgg/tf/myVGG',
                                                    save_best_only=True,
                                                    monitor='val_loss')]
    # tensorflow2.1 recommend to using fit
    history = model.fit(x=train_dataset,
                        steps_per_epoch=total_train // BATCH_SIZE,
                        epochs=epochs,
                        validation_data=validation_dataset,
                        validation_steps=total_val // BATCH_SIZE,
                        callbacks=callbacks)
    
    # plot loss and accuracy image
    history_dict = history.history
    train_loss = history_dict["loss"]
    train_accuracy = history_dict["accuracy"]
    val_loss = history_dict["val_loss"]
    val_accuracy = history_dict["val_accuracy"]

    # figure 1
    plt.figure()
    plt.plot(range(epochs), train_loss, label='train_loss')
    plt.plot(range(epochs), val_loss, label='val_loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')

    # figure 2
    plt.figure()
    plt.plot(range(epochs), train_accuracy, label='train_accuracy')
    plt.plot(range(epochs), val_accuracy, label='val_accuracy')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()
    
if __name__ == '__main__':
    main()
    
    
import os
import json
import sys

import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import GoogLeNet

def main():
    BATCH_SIZE = 32
    H,W = (224, 224)   
    
    data_root = os.path.abspath(os.getcwd())  # get data root path
    image_path = os.path.join(data_root, "data", "hymenoptera_data")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dir = os.path.join(image_path, "train")
    validation_dir = os.path.join(image_path, "val")
        
    epochs = 30
    
    train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=(H,W))

    validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                    shuffle=False,
                                                                    batch_size=BATCH_SIZE,
                                                                    image_size=(H,W))
    
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
    
    model = GoogLeNet(im_height=H, im_width=W, class_num=2, aux_logits=True)
    model.summary()
    
    
    # using keras low level api for training
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
    
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
    
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            aux1, aux2, output = model(images, training=True)
            loss1 = loss_fn(labels, aux1)
            loss2 = loss_fn(labels, aux2)
            loss3 = loss_fn(labels, output)
            loss = loss1 * 0.3 + loss2 * 0.3 + loss3
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        train_loss.update_state(loss)
        train_accuracy.update_state(labels, output)
    
    @tf.function
    def val_step(images, labels):
        _, _, output = model(images, training=False)
        loss = loss_fn(labels, output)

        val_loss.update_state(loss)
        val_accuracy.update_state(labels, output)
    
    
    best_val_acc = 0.
    for epoch in range(epochs):
        train_loss.reset_states()  # clear history info
        train_accuracy.reset_states()  # clear history info
        val_loss.reset_states()  # clear history info
        val_accuracy.reset_states()  # clear history info

        # train
        train_bar = tqdm(range(total_train // BATCH_SIZE), file=sys.stdout)
        for step, (images, labels) in zip(train_bar, train_dataset):
            train_step(images, labels)

            # print train process
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
                                                                                 epochs,
                                                                                 train_loss.result(),
                                                                                 train_accuracy.result())
        # validate
        val_bar = tqdm(range(total_val // BATCH_SIZE), file=sys.stdout)
        for step, (val_images, val_labels) in zip(val_bar, validation_dataset):
            val_step(val_images, val_labels)

            # print val process
            val_bar.desc = "valid epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
                                                                               epochs,
                                                                               val_loss.result(),
                                                                               val_accuracy.result())
        # only save best weights
        if val_accuracy.result() > best_val_acc:
            best_val_acc = val_accuracy.result()
            model.save("03_googlenet/tf/myGoogLeNet")

    
if __name__ == '__main__':
    main()
    
    
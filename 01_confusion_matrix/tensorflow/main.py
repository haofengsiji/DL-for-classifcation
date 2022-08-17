from logging.config import valid_ident
import os
import json
import pathlib
import math

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import tensorflow as tf

class ConfusionMatrix:
    
    def __init__(self, num_classes: int, labels: int):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1
            
    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall])
        print(table)
    
    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()
        
if __name__ == '__main__':
    batch_size = 4
    img_height = 224
    img_width = 224
    
    data_dir = "data/hymenoptera_data"
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
                os.path.join(data_dir, "val"),
                image_size=(img_height, img_width),
                batch_size=batch_size,
                shuffle=False)
    total_val = len(val_ds.file_paths)
    class_names = val_ds.class_names
    
    # load pretrain weights
    model_path = "/home/zhengfang/code/DL_CLS/00_hello_tensorflow/mn2_hym"
    model = tf.keras.models.load_model(model_path)
    
    # read class_indict
    labels = class_names
    confusion = ConfusionMatrix(num_classes=len(class_names), labels=labels)
    
    # validate
    for val_data in tqdm(iter(val_ds)):
        val_images, val_labels = val_data
        outputs = model(val_images, training=False)
        outputs = tf.argmax(outputs, axis=-1)
        confusion.update(outputs.numpy(), val_labels.numpy())
    confusion.plot()
    confusion.summary()
    
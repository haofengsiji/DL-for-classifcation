from tensorflow.keras import layers, Model, Sequential

def VGG(features, input_shape, num_classes=1000):
    # tensorflow中的tensor通道排序是NHWC
    input_image = layers.Input(shape=input_shape, dtype="float32")
    x = features(input_image)
    x = layers.Flatten()(x)
    x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dense(num_classes)(x)
    output = layers.Softmax()(x)
    model = Model(inputs=input_image, outputs=output)
    return model

def make_layers(cfg, batch_norm):
    feature_layers = []
    for v in cfg:
        if v == "M":
            feature_layers.append(layers.MaxPool2D())
        else:
            conv2d = layers.Conv2D(v, kernel_size=3, padding="SAME")
            if batch_norm:
                feature_layers += [conv2d, layers.BatchNormalization(), layers.ReLU()]
            else:
                feature_layers += [conv2d, layers.ReLU()]
            feature_layers.append(conv2d)
    return Sequential(feature_layers, name='features')

cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg(cfg="vgg16", input_shape=(224,224,3), num_classes=1000, batch_norm=True):
    assert cfg in cfgs.keys(), "not support model {}".format(cfg)
    model = VGG(make_layers(cfgs[cfg], batch_norm), input_shape=input_shape, num_classes=num_classes)
    return model
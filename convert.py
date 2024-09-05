import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Input, LeakyReLU, ZeroPadding2D, BatchNormalization, MaxPool2D
import cv2

def load_darknet_weights(model, weights_file):
    with open(weights_file, 'rb') as wf:
        major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)
        weights = np.fromfile(wf, dtype=np.float32)

    layers = model.layers
    for i, layer in enumerate(layers):
        if isinstance(layer, Conv2D):
            if i > 0:
                conv_layer = model.get_layer(index=i)
                filters = conv_layer.filters
                k_size = conv_layer.kernel_size[0]
                in_dim = conv_layer.input_shape[-1]
            else:
                conv_layer = model.get_layer(index=1)
                filters = conv_layer.filters
                k_size = conv_layer.kernel_size[0]
                in_dim = conv_layer.input_shape[-1]

            if i > 0:
                bn_layer = model.get_layer(index=i - 1)
            else:
                bn_layer = model.get_layer(index=0)

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, k_size, k_size)
            bn_shape = (filters,)

            # tf shape (height, width, in_dim, out_dim)
            conv_weights = weights[:np.prod(conv_shape)].reshape(conv_shape).transpose([2, 3, 1, 0])
            weights = weights[np.prod(conv_shape):]

            bn_weights = weights[:np.prod(bn_shape)]
            weights = weights[np.prod(bn_shape):]

            conv_layer.set_weights([conv_weights])
            bn_layer.set_weights(bn_weights)

    assert len(weights) == 0

def create_yolov4_model():
    input_layer = Input(shape=(416, 416, 3))

    # Your YOLOv4-tiny architecture here
    # This is a simplified example and may need adjustments
    x = Conv2D(32, 3, strides=(2, 2), padding='same', use_bias=False)(input_layer)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    # ... Add more layers according to your yolov4-tiny.cfg ...

    return Model(inputs=input_layer, outputs=x)

# Create model
model = create_yolov4_model()

# Load weights
load_darknet_weights(model, 'dashboard/model/dnn_model/yolov4-tiny.weights')

# Save as TensorFlow SavedModel
model.save('yolov4-tiny_savedmodel', save_format='tf')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model('yolov4-tiny_savedmodel')
tflite_model = converter.convert()

# Save TFLite model
with open('yolov4-tiny.tflite', 'wb') as f:
    f.write(tflite_model)

print("Conversion complete. TFLite model saved as 'yolov4-tiny.tflite'")
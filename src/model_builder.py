import os
import tensorflow as tf
from tensorflow.keras import layers, models

# Ensure Keras looks inside workspace .keras directory
os.environ["KERAS_HOME"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".keras"))

def _get_local_weights(filename):
    workspace_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".keras", "models", filename))
    if os.path.exists(workspace_path):
        return workspace_path
    return 'imagenet'

def build_custom_cnn(input_shape=(224, 224, 3), num_classes=44, dropout_rate=0.4):
    """
    Builds a custom 4-stage Convolutional Neural Network from scratch.
    Each stage consists of Conv2D -> BatchNormalization -> ReLU -> MaxPooling2D.
    Classification head contains GlobalAveragePooling2D -> Dense(256) -> BatchNorm -> Dropout -> Softmax.
    """
    inputs = layers.Input(shape=input_shape, name="input_tensor")
    
    # Block 1
    x = layers.Conv2D(32, (3, 3), padding='same', name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.ReLU(name='relu1')(x)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)
    
    # Block 2
    x = layers.Conv2D(64, (3, 3), padding='same', name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.ReLU(name='relu2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)
    
    # Block 3
    x = layers.Conv2D(128, (3, 3), padding='same', name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.ReLU(name='relu3')(x)
    x = layers.MaxPooling2D((2, 2), name='pool3')(x)
    
    # Block 4
    x = layers.Conv2D(256, (3, 3), padding='same', name='conv4')(x)
    x = layers.BatchNormalization(name='bn4')(x)
    x = layers.ReLU(name='relu4')(x)
    x = layers.MaxPooling2D((2, 2), name='pool4')(x)
    
    # Classification Head
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dense(256, activation='relu', name='dense_features')(x)
    x = layers.BatchNormalization(name='bn_head')(x)
    x = layers.Dropout(dropout_rate, name='dropout_head')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='prediction_output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="Custom_CNN_Baseline")
    return model

def build_vgg16(input_shape=(224, 224, 3), num_classes=44, dropout_rate=0.4, freeze_backbone=True):
    """
    Builds a Transfer Learning model using pre-trained VGG16 backbone.
    """
    w = _get_local_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    base_model = tf.keras.applications.VGG16(
        include_top=False,
        weights=w,
        input_shape=input_shape
    )
    base_model.trainable = not freeze_backbone
    
    inputs = layers.Input(shape=input_shape, name="vgg16_input")
    # Preprocessing expected by VGG16: scaled to 0-255 caffe format, or handled via keras preprocess_input
    # Since inputs in our data_loader are normalized [0, 1], we map to [0, 255] then vgg16 preprocess
    x = layers.Lambda(lambda z: z * 255.0)(inputs)
    x = layers.Lambda(tf.keras.applications.vgg16.preprocess_input)(x)
    x = base_model(x, training=False)
    
    x = layers.GlobalAveragePooling2D(name='vgg16_gap')(x)
    x = layers.Dense(256, activation='relu', name='vgg16_dense')(x)
    x = layers.BatchNormalization(name='vgg16_bn')(x)
    x = layers.Dropout(dropout_rate, name='vgg16_dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='vgg16_output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="VGG16_Transfer")
    return model, base_model

def build_resnet50(input_shape=(224, 224, 3), num_classes=44, dropout_rate=0.4, freeze_backbone=True):
    """
    Builds a Transfer Learning model using pre-trained ResNet50 backbone.
    """
    w = _get_local_weights('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights=w,
        input_shape=input_shape
    )
    base_model.trainable = not freeze_backbone
    
    inputs = layers.Input(shape=input_shape, name="resnet50_input")
    x = layers.Lambda(lambda z: z * 255.0)(inputs)
    x = layers.Lambda(tf.keras.applications.resnet50.preprocess_input)(x)
    x = base_model(x, training=False)
    
    x = layers.GlobalAveragePooling2D(name='resnet_gap')(x)
    x = layers.Dense(256, activation='relu', name='resnet_dense')(x)
    x = layers.BatchNormalization(name='resnet_bn')(x)
    x = layers.Dropout(dropout_rate, name='resnet_dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='resnet_output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="ResNet50_Transfer")
    return model, base_model

def build_mobilenet_v2(input_shape=(224, 224, 3), num_classes=44, dropout_rate=0.3, freeze_backbone=True):
    """
    Builds an edge-optimized Transfer Learning model using pre-trained MobileNetV2 backbone.
    """
    w = _get_local_weights('mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5')
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights=w,
        input_shape=input_shape
    )
    base_model.trainable = not freeze_backbone
    
    inputs = layers.Input(shape=input_shape, name="mobilenetv2_input")
    # MobileNetV2 expects input in [-1, 1], since input is [0, 1]: (z * 2.0) - 1.0
    x = layers.Lambda(lambda z: (z * 2.0) - 1.0)(inputs)
    x = base_model(x, training=False)
    
    x = layers.GlobalAveragePooling2D(name='mobilenet_gap')(x)
    x = layers.Dropout(dropout_rate, name='mobilenet_dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='mobilenet_output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_Transfer")
    return model, base_model

def count_parameters(model):
    """
    Returns total, trainable, and non-trainable parameter counts.
    """
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": non_trainable_params
    }

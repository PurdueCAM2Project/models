"""Mesh RCNN models."""
import tensorflow as tf

class MeshRCNNModel(tf.keras.Model):
    """Mesh RCNN Model definition"""
    def __init__(self, 
                 backbone: tf.keras.Model,
                 decoder: tf.keras.Model,
                 **kwargs):

                 
        super(MeshRCNNModel, self).__init__(**kwargs)
        


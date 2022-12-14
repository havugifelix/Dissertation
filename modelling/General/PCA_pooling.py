
from tensorflow.keras.layers import *
import tensorflow as tf
class pca(Layer):
    def __init__(self, n_components = 1):
        super().__init__()
        self.g = None
        self.n_components = n_components
     
        
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'g': self.g,
            'n_components': self.n_components
        })
        return config
    
    @tf.function
    def call(self, X):
        
        m, p, q, n = X.shape
        X_r = tf.transpose(X, perm=[0,3,1,2])
        X_r = tf.reshape(X_r, (-1, p, q))
        if m != None:
            i=0
            if self.g is None:
                self.g = tf.Variable(tf.zeros((m*n, p, self.n_components)))
            
            for x in X_r:
                
                cov = tf.tensordot(x, x, axes=1)
                s, u, v = tf.linalg.svd(cov)
                U1 = u[:, 0:self.n_components]

                x_proj = tf.tensordot(x, U1, axes=1)
                
                self.g[i].assign(x_proj)
                i += 1
            return tf.reshape(self.g, (m, -1, n))
        
        return X[:, :, 0, :]
class pca_mean(Layer):
    def __init__(self, n_components = 1):
        super().__init__()
        self.g = None
        self.n_components = n_components
        
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'g': self.g,
            'n_components': self.n_components
        })
        return config
    
    @tf.function
    def call(self, X):
        m, p, q, n = X.shape
        if m != None:
            i=0
            if self.g is None:
                pca_layer = pca()
                
                self.g = pca_layer(X)
                return tf.reduce_mean(self.g, axis=1)
        
        return X[:, 0, 0, :]

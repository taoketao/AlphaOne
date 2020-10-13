''' a loss function '''

#rom __future__ import absolute_import
#import six
from keras import backend as K
#from keras.utils.generic_utils import deserialize_keras_object
#from keras.utils.generic_utils import serialize_keras_object

def huber(y_true, y_pred, delta):
    err=y_true - y_pred
    abs_err = K.maximum( err,-err )
    if abs_err < delta:
        return 0.5*abs_err**2
    return delta * (abs_err - 0.5*delta)


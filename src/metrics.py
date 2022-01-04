from tensorflow.keras import backend as K

def distance(y_true, y_pred):
    x_true = K.flatten(K.argmax(y_true, axis=1)) 
    x_pred = K.flatten(K.argmax(y_pred, axis=1))

    # measure distance for crops with vertebrae only
    pos = K.cast(K.sum(y_true, axis=(1, 2)) > 0, 'float32')
    dist = K.cast(x_true - x_pred, 'float32') # diff in mm
    
    return K.abs(pos * dist) # (batch_size,) shape

def pos_conf(y_true, y_pred):
    x_true = K.flatten(K.argmax(y_true, axis=1)) 
    x_pred = K.flatten(K.argmax(y_pred, axis=1))

    # prediction confidences for positive samples
    pos_idx = K.cast(K.sum(y_true, axis=(1, 2)) > 0, 'float32')
    conf = K.cast(K.flatten(K.max(y_pred, axis=1)), 'float32') * pos_idx
    
    return conf # (batch_size,) shape

def neg_conf(y_true, y_pred):
    x_true = K.flatten(K.argmax(y_true, axis=1)) 
    x_pred = K.flatten(K.argmax(y_pred, axis=1))

    # prediction confidences for negative samples 
    neg_idx = K.cast(K.sum(y_true, axis=(1, 2)) == 0, 'float32')
    conf = K.cast(K.flatten(K.max(y_pred, axis=1)), 'float32') * neg_idx
    
    return conf # (batch_size,) shape
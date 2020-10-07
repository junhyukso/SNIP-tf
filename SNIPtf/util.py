import numpy as np

def calc_weights_sparsity(model):
    all_weights =  np.concatenate([ w.numpy().flatten() for w in model.trainable_variables]  )
    sparsity    =  (1 - np.count_nonzero(all_weights)/all_weights.size)*100
    return sparsity

# SNIP-tf
tf-keras implementation of SNIP(Single-shot Network Pruning based on Connection Sensitivity)

## Usage
```python
import tensorflow as tf
from SNIPtf.prune import make_prune_callback
...
(x_train,y_train) , (x_test,y_test) = ...

model = ...

# Prune 90% weight
# Feed Mini Batch to make pruning callback.
pc = make_prune_callback( 
  model,                                # model instance
  0.9  ,                                # Sparsity. 0~1 float
  tf.convert_to_tensor(x_train[1:10]) , # Feed Mini Batch. X
  tf.convert_to_tensor(y_train[1:10])   # Feed Mini Batch. Y
  )

# Train with pruning callback
model.fit( ...
           callbacks = [pc , ..]
          )
```
or see example.py

## Note
This Implementation currently not reduce Train FLOPS.

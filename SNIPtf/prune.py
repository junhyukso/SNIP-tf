import tensorflow as tf

class Prune_Callback(tf.keras.callbacks.Callback):
  def __init__(self, masks):
    super(Prune_Callback, self).__init__()
    self.masks = masks

  def _prune(self, model) :
    for var, mask in list(zip(self.model.trainable_variables , self.masks)) :
      var.assign(
        tf.math.multiply(var.read_value() , mask)
      )

  def on_train_batch_begin(self, batch, logs=None):
    self._prune(self.model)

  def on_train_batch_end(self, batch, logs=None):
    self._prune(self.model)


def make_prune_callback(model, sparsity, x, y) :
  with tf.GradientTape() as tape:
    y_pred  = model(x)
    loss = model.compiled_loss(y,y_pred)    

  grads = tape.gradient(loss,model.trainable_variables)
  saliences = [ tf.abs(grad*weight) for weight, grad in zip(model.trainable_variables, grads) ]
  
  saliences_flat = tf.concat( [ tf.reshape(s,-1) for s in saliences ] ,0)
  
  k = tf.dtypes.cast(
          tf.math.round(
              tf.dtypes.cast(tf.size(saliences_flat), tf.float32) *
              (1 - sparsity)), tf.int32)
  #print(k,"/",tf.size(saliences_flat))
  values, _ = tf.math.top_k(
          saliences_flat, k=tf.size(saliences_flat)
  )
  current_threshold = tf.gather(values, k - 1)
  #print(current_threshold)
  masks = [ tf.cast(tf.greater_equal(s,current_threshold),dtype=s.dtype ) for s in saliences ]

  return Prune_Callback(masks)

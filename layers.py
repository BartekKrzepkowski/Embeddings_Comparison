from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import tensorflow_hub as hub
import tensorflow as tf

class OneHotLayer(Layer):
    def __init__(self, input_dim, **kwargs):
        self.input_dim = input_dim
        super(OneHotLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(OneHotLayer, self).build(input_shape)

    def call(self, x, mask=None):
        return tf.one_hot(indices=tf.cast(x, dtype=tf.int32), depth=self.input_dim)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.input_dim)

    
    
class ElmoLayer(Layer):
    def __init__(self, trainable=True, dict_output="default", signature="default", **kwargs):
        self.output_size = 1024
        self.signature = signature
        self.dict_output = dict_output
        self.trainable=trainable
        super(ElmoLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module(
            'https://tfhub.dev/google/elmo/2',
            trainable=self.trainable,
            name="{}_module".format(self.name)
        )
        if self.trainable:
            self._trainable_weights.extend(tf.trainable_variables(scope="^{}_module/.*".format(self.name)))
            
        super(ElmoLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(
            tf.squeeze(tf.cast(x, dtype=tf.string), axis=1),
            as_dict=True,
            signature=self.signature
        )[self.dict_output]
        return result

    def compute_output_shape(self, input_shape):
        if dict_output == "default":
            return (input_shape[0], self.output_size)
        else:
            return (input_shape[0], None, self.output_size)
    
    
    
# dict_output
# Use "pooled_output" for classification tasks on an entire sentence.
# Use "sequence_outputs" for token-level output.
# signature
# Use "tokens", because only that option is currently supported(04.2019), assumes pre-processed inputs
class BertLayer(Layer):
    def __init__(self, trainable=True, dict_output="pooled_output", signature="tokens", n_fine_tune_layers=3, **kwargs):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.output_size = 768
        self.signature = signature
        self.dict_output = dict_output
        self.trainable=trainable
        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(
            "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
            trainable=self.trainable,
            name="{}_module".format(self.name)
        )
        trainable_vars = self.bert.variables
        
        # Remove unused layers
        trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]
        
        # Select how many layers to fine tune
        trainable_vars = trainable_vars[-self.n_fine_tune_layers :]
        
        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)
        
        # Add non-trainable weights
        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)
        
        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [tf.cast(x, dtype=tf.string) for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids
        )
        result = self.bert(
            inputs=bert_inputs,
            as_dict=True,
            signature=self.signature
        )[self.dict_output]
        return result

    def compute_output_shape(self, input_shape):
        if dict_output == "pooled_output":
            return (input_shape[0], self.output_size)
        else:
            return (input_shape[0], None, self.output_size)
        
        
        
def BertInputLayer(Layer):
    def __init__(self, **kwargs):
        
        super(BertInputLayer, self).__init__(**kwargs)
        
    def build(self, input_shape, dtype):
        self.input_id = Input(shape=input_shape, dtype=dtype, name="input_id")
        self.input_mask = Input(shape=input_shape, dtype=dtype, name="input_mask")
        self.input_segment = Input(shape=input_shape, dtype=dtype, name="input_segment")
        super(BertInputLayer, self).build(input_shape)
    
    def call(self, x, mask=None):
        return [self.input_id, self.input_mask, self.input_segment]

    
    
    
class AttensionLayer(Layer):
    def __init__(self, trainable=True):
        pass
    

    
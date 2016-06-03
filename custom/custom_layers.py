from keras import backend as K
from keras.engine.topology import Layer
from keras.engine import InputSpec
import numpy as np

class MaskedConcat(Layer):
    """
    Masked concats 2 3D tensors along -1 axis
    Preserves masking
    """

    def __init__(self, concat_axis, layers=None, **kwargs):
        self.concat_axis = concat_axis
        self.supports_masking = True
        super(MaskedConcat, self).__init__(**kwargs)

        if layers:
            node_indices = [0 for _ in range(len(layers))]
            self.built = True
            self.add_inbound_node(layers, node_indices, None)
        else:
            self.built = False

    def call(self, inputs, mask=None):
        assert isinstance(inputs, list)
        if mask:
            assert isinstance(mask, list) or isinstance(mask, tuple)
            if self.concat_axis == 2:
                assert K.equal(mask[0], mask[1])

        return K.concatenate(inputs, axis=self.concat_axis)

    def compute_mask(self, inputs, input_mask=None):
        if input_mask:
            assert(isinstance(input_mask, list) or isinstance(input_mask, tuple))
            if self.concat_axis == 2:
                assert(K.equal(input_mask[0], input_mask[1]))
                return input_mask[0]
            elif self.concat_axis == 1:
                return K.concatenate(input_mask, axis=1)
            else:
                return None
        else:
            return None

    def get_output_shape_for(self, input_shapes):
        if self.concat_axis == 2:
            in_shape = input_shapes[0]
            output_shape = (in_shape[0], in_shape[1], in_shape[2] * 2)
        elif self.concat_axis == 1:
            output_shape = (input_shapes[0][0], input_shapes[0][1] + input_shapes[1][1], input_shapes[0][2])
        return output_shape

def masked_concat(inputs, concat_axis=2):
    """
    Functional API for MaskedConcat
    """
    concat_layer = MaskedConcat(concat_axis=concat_axis)
    return concat_layer(inputs)

class Reverse(Layer):
    """
    Reverses 3D tensor along axis 1 (axis representing input length)

    Mask should be a 2D tensor
    """
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(Reverse, self).__init__(**kwargs)

    def compute_mask(self, x, input_mask=None):
        if input_mask:
            assert K.ndim(input_mask) == 2 #must be 2D
            return input_mask[:, ::-1]
        else:
            return None

    def call(self, x, mask=None):
        assert (K.ndim(x) == 3) #3D tensor
        return x[:,::-1,:]

    def get_output_shape_for(self, input_shape):
        return input_shape

class MaskedDot(Layer):
    """
    same a merge with dot on axis=(1,1) but it supports masking
    """
    def __init__(self, layers=None, **kwargs):
        self.supports_masking = True
        super(MaskedDot, self).__init__(**kwargs)

        if layers:
            node_indices = [0 for _ in range(len(layers))]
            self.built = True
            self.add_inbound_node(layers, node_indices, None)
        else:
            self.built = False

    def call(self, inputs, mask=None):
        """
        just does batch dot
        """
        assert K.equal(mask[0], mask[1])

        return K.batch_dot(inputs[0], inputs[1], axes=(1,1))

    def compute_mask(self, inputs, input_mask=None):
        return None

    def get_output_shape_for(self, input_shapes):
        input_one = input_shapes[0]
        input_two = input_shapes[1]
        return (input_one[0], input_two[2])

def masked_dot(inputs):
    masked_dot_layer = MaskedDot()
    return masked_dot_layer(inputs)

class MaskedSum(Layer):
    """
    Sums 2 3D tensors, accounts for mask
    """
    def __init__(self, layers=None, **kwargs):
        self.supports_masking = True
        super(MaskedSum, self).__init__(**kwargs)

        if layers:
            node_indices = [0 for _ in range(len(layers))]
            self.built = True
            self.add_inbound_node(layers, node_indices, None)
        else:
            self.built = False

    def call(self, inputs, mask=None):
        """
        Mask 1 exists, while mask 2 does not
        """
        assert isinstance(mask, list)
        assert mask[0] != None and mask[1] == None

        return inputs[0] + (inputs[1] * K.expand_dims(mask[0]))

    def compute_mask(self, inputs, input_mask=None):
        return input_mask[0]

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

def masked_sum(inputs):
    masked_sum_layer = MaskedSum()
    return masked_sum_layer(inputs)

class MaskedFlatten(Layer):
    def __init__(self, **kwargs):
        self.input_spec = [InputSpec(ndim='3+')]
        self.supports_masking = True
        super(MaskedFlatten, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return K.batch_flatten(x)

    def compute_mask(self, input, input_mask=None):
        if input_mask:
            return input_mask
        else:
            return None

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], np.prod(input_shape[1:]))

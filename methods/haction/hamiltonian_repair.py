"""Experimental, explicit Hf operator; not an official DIVA implementation.

The upstream dynamic kernel has ambiguous batch/channel axes. This version does
not guess that layout. It uses the discrete Schroedinger action
Hf = alpha * (-Laplacian(f)) + (V + I) * f, channel by channel.
It implements a discrete Schroedinger operator on a unit grid, but does not
reproduce DIVA equations (5)-(6)'s patch-dependent convolutional projection.
The CNN below additionally chooses V=features and learns I from feature maps;
those are experimental choices, not the paper's explicit patch interactions.
"""
import tensorflow as tf
from tensorflow.keras import layers as L


@tf.keras.utils.register_keras_serializable(package='DIVAResearch')
class HamiltonianAction(L.Layer):
    """Apply Hf = alpha*(-Delta)f + (V+I)*f independently per channel.

    Inputs are three equal-shape real NHWC tensors [f, V, I]. For fixed V and I,
    this is a symmetric linear operator; the complete map is nonlinear when
    V or I depends on f. Alpha is a spatially constant learned scalar per
    channel, parametrized by softplus. Grid spacing is one pixel.

    ``dirichlet`` uses zero-valued exterior samples (the historical default).
    ``neumann`` duplicates edge samples and matches the degree-diagonal graph
    Laplacian in the author's De-QuIP f_Hamil2D.m (for square patches n>=2).
    Neither option computes an eigenbasis or DIVA's adaptive projection kernel.
    """

    def __init__(self, boundary='dirichlet', **kwargs):
        super().__init__(**kwargs)
        if boundary not in ('dirichlet', 'neumann'):
            raise ValueError("boundary must be 'dirichlet' or 'neumann'")
        self.boundary = boundary

    def build(self, input_shapes):
        if len(input_shapes) != 3:
            raise ValueError('HamiltonianAction requires [features, potential, interaction]')
        shapes = [tf.TensorShape(shape) for shape in input_shapes]
        if any(shape.rank != 4 for shape in shapes):
            raise ValueError('All HamiltonianAction inputs must be rank-4 NHWC tensors')
        if not all(shapes[0].is_compatible_with(shape) for shape in shapes[1:]):
            raise ValueError('features, potential and interaction must have identical shapes')
        if shapes[0][-1] is None:
            raise ValueError('The channel dimension must be known at build time')
        channels=int(shapes[0][-1])
        self.input_spec = [L.InputSpec(ndim=4, axes={-1: channels}) for _ in range(3)]
        self.raw_alpha=self.add_weight('raw_alpha',shape=(channels,),initializer=tf.keras.initializers.Constant(-4.6))
        super().build(input_shapes)

    def call(self, inputs):
        features,potential,interaction=inputs
        checks = [tf.debugging.assert_equal(tf.shape(features), tf.shape(term),
                  message='HamiltonianAction forbids broadcasting between inputs')
                  for term in (potential, interaction)]
        # Build a fixed stencil in the active dtype/graph, without adding weights.
        stencil=tf.constant([[0.,-1.,0.],[-1.,4.,-1.],[0.,-1.,0.]],dtype=features.dtype)
        stencil=tf.tile(stencil[:,:,None,None],[1,1,int(self.raw_alpha.shape[0]),1])
        with tf.control_dependencies(checks):
            if self.boundary == 'neumann':
                padded=tf.pad(features,[[0,0],[1,1],[1,1],[0,0]],mode='SYMMETRIC')
                kinetic=tf.nn.depthwise_conv2d(padded,stencil,[1,1,1,1],padding='VALID')
            else:
                kinetic=tf.nn.depthwise_conv2d(features,stencil,[1,1,1,1],padding='SAME')
            return tf.nn.softplus(self.raw_alpha)*kinetic+(potential+interaction)*features

    def get_config(self):
        return {**super().get_config(), 'boundary': self.boundary}


def repaired_model(depth=10,filters=64,kernel_size=5):
    image=L.Input((None,None,1),name='input0')
    features=L.Conv2D(filters,kernel_size,padding='same',kernel_initializer='Orthogonal',name='initial_patches')(image)
    features=L.Activation('relu',name='initial_patch_acti')(features)
    interaction=L.Conv2D(filters,kernel_size,padding='same',kernel_initializer='Orthogonal',name='interactions')(features)
    interaction=L.Activation('relu',name='interaction_acti0')(interaction)
    x=HamiltonianAction(name='proj_coef')([features,features,interaction])
    count=0
    for _ in range(depth-2):
        count+=1
        x=L.Conv2D(filters,kernel_size,padding='same',use_bias=False,kernel_initializer='Orthogonal',name=f'conv{count}')(x)
        count+=1
        x=L.BatchNormalization(axis=3,momentum=.1,epsilon=.0001,name=f'bn{count}')(x)
        x=L.Activation('relu',name=f'Thresholding{count}')(x)
    residual=L.Conv2D(1,kernel_size,padding='same',use_bias=False,kernel_initializer='Orthogonal',name='inv_trans')(x)
    output=L.Subtract(name='subtract')([image,residual])
    return tf.keras.Model(image,output,name='DIVA_experimental_H_action')


def original_model(depth=10,filters=64,kernel_size=5):
    from pathlib import Path
    import json
    from types import SimpleNamespace
    from tensorflow.python.keras.utils import conv_utils
    p=Path(__file__).resolve().parents[1] / 'diva/upstream/DIVA_training.ipynb'
    source=''.join(json.loads(p.read_text(encoding='utf-8'))['cells'][0]['source'])
    source=source[source.index('class Hamiltonian_Conv2D'):source.index('def findLastCheckpoint')]
    scope={'tf':tf,'K':tf.keras.backend,'conv_utils':conv_utils,'args':SimpleNamespace(kernel_size=kernel_size),'Model':tf.keras.Model}
    scope.update({name:getattr(L,name) for name in ['Input','Conv2D','BatchNormalization','Activation','Subtract']})
    # Only the unmodified class and builder definitions are executed.
    exec(compile(source,'upstream_definitions.py','exec'),scope)
    return scope['DIVA2D'](depth=depth,filters=filters,kernel_size=kernel_size)

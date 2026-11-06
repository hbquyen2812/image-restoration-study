"""Independent matrix, gradient and compatibility audit; no model training.

Run with the DIVA TensorFlow 2.10 environment. References and qualifications are
in outputs/diva_research/HAMILTONIAN_REVIEW_VI.md. This checks an experimental
Schroedinger action, not a reproduction of DIVA's projection architecture.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from hamiltonian_repair import HamiltonianAction, repaired_model, original_model

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(1)
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/diva_research/experiments/hamiltonian_review'
OUT.mkdir(parents=True, exist_ok=True)


def negative_laplacian(height, width, boundary):
    """Explicit grid adjacency; independent of convolution implementation."""
    matrix = np.zeros((height * width, height * width), np.float64)
    for r in range(height):
        for c in range(width):
            k = r * width + c
            degree = 0
            for rr, cc in [(r-1,c), (r+1,c), (r,c-1), (r,c+1)]:
                if 0 <= rr < height and 0 <= cc < width:
                    matrix[k, rr * width + cc] = -1
                    degree += 1
            matrix[k, k] = 4 if boundary == 'dirichlet' else degree
    return matrix


def matlab_boundary_construction(n):
    """Audit translation of f_Hamil2D.m kinetic matrix for poids=1, V=0.

    Not executed in MATLAB; comparing its explicitly translated indexing rules.
    Source pinned to De-QuIP commit 384e24f217ea71352fd48ff715fb84b0c4b487f8.
    """
    size = n * n
    h = 4 * np.eye(size)
    for offset in (1, n):
        h -= np.diag(np.ones(size-offset), offset)
        h -= np.diag(np.ones(size-offset), -offset)
    for block in range(n):
        h[n*block,n*block] -= 1
        h[n*(block+1)-1,n*(block+1)-1] -= 1
    for i in range(n):
        h[i,i] -= 1
        h[size-1-i,size-1-i] -= 1
    for i in range(1,n):
        h[n*i,n*i-1] = h[n*i-1,n*i] = 0
    return h


def main():
    rng = np.random.default_rng(123)
    rows = []
    for boundary in ('dirichlet', 'neumann'):
        for height, width in [(1,1), (1,4), (2,3), (3,3), (4,5)]:
            shape = (2,height,width,2)
            x,v,interaction = [rng.normal(size=shape) for _ in range(3)]
            layer = HamiltonianAction(boundary=boundary, dtype='float64')
            layer([x,v,interaction])
            layer.raw_alpha.assign([-1.2,.4])
            alpha = np.logaddexp(0,layer.raw_alpha.numpy())
            lap = negative_laplacian(height,width,boundary)
            expected = np.empty_like(x)
            symmetry_error = 0.
            for b in range(2):
                for c in range(2):
                    h = alpha[c]*lap + np.diag((v+interaction)[b,:,:,c].ravel())
                    expected[b,:,:,c] = (h @ x[b,:,:,c].ravel()).reshape(height,width)
                    symmetry_error = max(symmetry_error,float(np.max(np.abs(h-h.T))))
            actual = layer([x,v,interaction]).numpy()
            error = float(np.max(np.abs(actual-expected)))
            assert error < 1e-12
            u = rng.normal(size=shape)
            hu = layer([u,v,interaction]).numpy()
            adjoint_error = float(abs(np.sum(u*actual)-np.sum(hu*x)))
            assert adjoint_error < 1e-11
            eigen_min = float(np.linalg.eigvalsh(lap).min())
            assert eigen_min > -1e-12
            rows.append(dict(boundary=boundary,shape=list(shape),matrix_error=error,
                             symmetry_error=symmetry_error,adjoint_error=adjoint_error,
                             kinetic_min_eigenvalue=eigen_min))

    matlab_errors = {str(n):float(np.max(np.abs(matlab_boundary_construction(n)-
                      negative_laplacian(n,n,'neumann')))) for n in (2,3,5,15)}
    assert max(matlab_errors.values()) == 0

    # Central numerical differentiation independently checks all three inputs.
    x,v,interaction = [tf.constant(rng.normal(size=(1,2,3,2)),tf.float64) for _ in range(3)]
    gradients = {}
    for boundary in ('dirichlet','neumann'):
        layer = HamiltonianAction(boundary=boundary,dtype='float64')
        layer([x,v,interaction]); layer.raw_alpha.assign([-.7,.3])
        theory,numerical = tf.test.compute_gradient(lambda a,b,c: layer([a,b,c]),
                                                     [x,v,interaction],delta=1e-5)
        errors = [float(np.max(np.abs(a-b))) for a,b in zip(theory,numerical)]
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(layer([x,v,interaction])**2)
        analytical = tape.gradient(loss,layer.raw_alpha).numpy()
        initial = layer.raw_alpha.numpy().copy()
        numerical_alpha = []
        for i in range(2):
            delta = np.zeros(2); delta[i] = 1e-5
            layer.raw_alpha.assign(initial+delta)
            upper = float(tf.reduce_sum(layer([x,v,interaction])**2))
            layer.raw_alpha.assign(initial-delta)
            lower = float(tf.reduce_sum(layer([x,v,interaction])**2))
            numerical_alpha.append((upper-lower)/2e-5)
        layer.raw_alpha.assign(initial)
        alpha_error = float(np.max(np.abs(analytical-numerical_alpha)))
        assert max(errors) < 1e-8 and alpha_error < 1e-7
        gradients[boundary] = dict(input_jacobian_max_errors=errors,raw_alpha_error=alpha_error)

    dtype_rows = []
    for dtype in ('float32','float64','mixed_float16','mixed_bfloat16'):
        layer = HamiltonianAction(dtype=dtype)
        sample = tf.ones([2,4,5,2])
        with tf.GradientTape() as tape:
            y = layer([sample,sample,sample])
            loss = tf.reduce_sum(y*y)
        grad = tape.gradient(loss,layer.raw_alpha)
        assert bool(tf.reduce_all(tf.math.is_finite(y)))
        assert grad is not None and bool(tf.reduce_all(tf.math.is_finite(grad)))
        dtype_rows.append(dict(policy=dtype,output_dtype=y.dtype.name,status='passed'))

    rejects = []
    layer = HamiltonianAction()
    layer([tf.ones([2,4,5,2])]*3)
    for bad_shape in [(1,4,5,2),(2,1,5,2),(2,4,5,1)]:
        try:
            layer([tf.ones([2,4,5,2]),tf.ones(bad_shape),tf.ones([2,4,5,2])])
        except (ValueError,tf.errors.InvalidArgumentError): rejects.append(list(bad_shape))
        else: raise AssertionError(f'Silent broadcasting: {bad_shape}')

    graph_layer = HamiltonianAction()
    @tf.function(input_signature=[tf.TensorSpec([None,None,None,2],tf.float32)]*3)
    def graph(a,b,c):
        return graph_layer([a,b,c])
    graph(tf.ones([2,4,5,2]),tf.ones([2,4,5,2]),tf.ones([2,4,5,2]))
    try:
        graph(tf.ones([2,4,5,2]),tf.ones([1,4,5,2]),tf.ones([2,4,5,2]))
    except tf.errors.InvalidArgumentError: dynamic_shape_rejected = True
    else: raise AssertionError('Graph silently broadcast batch dimension')
    graph_layer([tf.ones([1,3,7,2])]*3)  # First build in tf.function, then eager.

    # Edge behavior, and isolation of independently processed batch/channels.
    constants = {}
    isolation = {}
    for boundary in ('dirichlet','neumann'):
        layer = HamiltonianAction(boundary=boundary,dtype='float64')
        one = np.ones((2,5,5,2)); zero = np.zeros_like(one)
        layer([one,zero,zero]); layer.raw_alpha.assign([0,0])
        y = layer([one,zero,zero]).numpy()/np.log(2)
        constants[boundary] = dict(center=float(y[0,2,2,0]),
                                  edge=float(y[0,0,2,0]),corner=float(y[0,0,0,0]))
        if boundary == 'neumann': assert np.max(np.abs(y)) == 0
        else: np.testing.assert_allclose([y[0,2,2,0],y[0,0,2,0],y[0,0,0,0]],[0,1,2])
        v2=zero.copy(); v2[0,:,:,0]=1
        delta=layer([one,v2,zero]).numpy()-layer([one,zero,zero]).numpy()
        assert np.max(np.abs(delta[1])) == 0 and np.max(np.abs(delta[:,:,:,1])) == 0
        isolation[boundary] = 'passed'
        ins=[tf.keras.Input((None,None,2),dtype='float64') for _ in range(3)]
        model=tf.keras.Model(ins,layer(ins))
        path=OUT/f'operator_{boundary}.h5'
        model.save(path,include_optimizer=False)
        restored=tf.keras.models.load_model(path,compile=False)
        np.testing.assert_array_equal(model([one,v2,zero]),restored([one,v2,zero]))

    # Optional historical compatibility check: old artifacts are not distributed.
    snapshot=OUT/'hamiltonian_repair_before_review.py'
    checkpoint=OUT.parent/'experimental_H_action_smoke.h5'
    sample=rng.random((1,40,40,1)).astype('float32')
    compatibility_error=None
    compatibility_status='skipped: historical source snapshot/checkpoint not installed'
    if snapshot.is_file() and checkpoint.is_file():
        spec=importlib.util.spec_from_file_location('old_h_action',snapshot)
        tf.keras.utils.get_custom_objects().pop('DIVAResearch>HamiltonianAction')
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        try:
            old_model=tf.keras.models.load_model(checkpoint,compile=False,
                          custom_objects={'DIVAResearch>HamiltonianAction':old.HamiltonianAction})
            tf.keras.utils.get_custom_objects()['DIVAResearch>HamiltonianAction']=HamiltonianAction
            new_model=tf.keras.models.load_model(checkpoint,compile=False,
                          custom_objects={'DIVAResearch>HamiltonianAction':HamiltonianAction})
            compatibility_error=float(np.max(np.abs(old_model(sample)-new_model(sample))))
            assert compatibility_error == 0
            assert isinstance(new_model.get_layer('proj_coef'),HamiltonianAction)
            compatibility_status='passed'
        finally:
            tf.keras.utils.get_custom_objects()['DIVAResearch>HamiltonianAction']=HamiltonianAction

    model=repaired_model(depth=4,filters=16)
    with tf.GradientTape() as tape:
        loss=tf.reduce_mean(model(sample,training=True)**2)
    grads=tape.gradient(loss,model.trainable_variables)
    assert all(g is not None and bool(tf.reduce_all(tf.math.is_finite(g))) for g in grads)

    original_stencil=np.array([[2,-1,0],[-1,4,-1],[0,-1,2]],dtype=float)
    free_weights=np.ones((3,3));free_weights[0,1]=2
    weighted=original_stencil*free_weights
    upstream=original_model(depth=4,filters=4)
    projection=upstream.get_layer('proj_coef')
    projection.kernel.assign(-tf.ones_like(projection.kernel))
    upstream_output=projection(tf.ones([1,8,8,4])).numpy()
    upstream_negative_count=int(np.sum(upstream_output<0))
    assert upstream_negative_count == upstream_output.size
    result=dict(status='passed',tensorflow=tf.__version__,seed=123,
        scope='H action operator audit, not DIVA reproduction or MATLAB execution',
        matrix_checks=rows,matlab_translation_neumann_errors=matlab_errors,
        numerical_gradients=gradients,dtypes=dtype_rows,rejected_shapes=rejects,
        dynamic_shape_rejected=dynamic_shape_rejected,constant_field_response_div_alpha=constants,
        batch_channel_isolation=isolation,serialization_both_boundaries='passed',
        previous_smoke_checkpoint_output_error=compatibility_error,
        historical_compatibility_status=compatibility_status,
        full_small_model_all_gradients_finite=True,full_small_model_params=model.count_params(),
        upstream_relu_argument_ignored=dict(negative_outputs=upstream_negative_count,
                                            output_count=int(upstream_output.size)),
        upstream_stencil_sum=float(original_stencil.sum()),
        counterexample_unconstrained_weights_break_centrosymmetry=bool(np.any(weighted!=weighted[::-1,::-1])))
    (OUT/'math_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    files=[Path(__file__),Path(__file__).with_name('hamiltonian_repair.py')]
    if snapshot.is_file(): files.append(snapshot)
    files+=list((ROOT/'outputs/diva_research/sources/DeQuIP_reference').glob('*.m'))
    (OUT/'source_checksums.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                                       for p in files},indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()

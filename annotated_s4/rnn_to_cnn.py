import jax
import jax.numpy as np
from network_architecture.annotated_s4.ssm_utils import random_SSM, discretize, scan_SSM, run_SSM

def K_conv(Ab, Bb, Cb, L):
    return np.array(
        [(Cb @ matrix_power(Ab, l) @ Bb).reshape() for l in range(L)]
    )

def causal_convolution(u, K, nofft=False):
    if nofft:
        return convolve(u, K, mode="full")[: u.shape[0]]
    else:
        assert K.shape[0] == u.shape[0], f"K:{K.shape}, u:{u.shape}"
        #ud = np.fft.rfft(np.pad(u, ((0, K.shape[0]),(0,0))), axis=0)
        ud = np.fft.rfft(np.pad(u, (0, K.shape[0])), axis=0)
        Kd = np.fft.rfft(np.pad(K, (0, u.shape[0])))
        #Kd_expanded = np.expand_dims(Kd, axis=1)  # Expand Kd to shape (785, 1)
        out = ud * Kd  # This should now correctly broadcast
        print('ud', ud.shape, 'Kd', Kd.shape, 'out', out.shape)
        iout =  np.fft.irfft(out, axis=0)
        return iout[: u.shape[0]]
    
def test_cnn_is_rnn(N=4, L=16, step=1.0 / 16):
    ssm = random_SSM(rng, N)
    u = jax.random.uniform(rng, (L,))
    jax.random.split(rng, 3)
    # RNN
    rec = run_SSM(*ssm, u)

    # CNN
    ssmb = discretize(*ssm, step=step)
    conv = causal_convolution(u, K_conv(*ssmb, L))

    # Check
    assert np.allclose(rec.ravel(), conv.ravel())

def log_step_initializer(dt_min=0.001, dt_max=0.1):
    def init(key, shape):
        return jax.random.uniform(key, shape) * (
            np.log(dt_max) - np.log(dt_min)
        ) + np.log(dt_min)

    return init
import jax.numpy as np
from flax import linen as nn
from annotated_s4.ssm_layer import SSMLayer
from annotated_s4.stacked_model import StackedModel

def cloneLayer(layer):
    return nn.vmap(
        layer,
        in_axes=1,
        out_axes=1,
        variable_axes={"params": 1, "cache": 1, "prime": 1},
        split_rngs={"params": True},
    )


def make_HiPPO(N):
    P = np.sqrt(1 + 2 * np.arange(N))
    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = np.tril(A) - np.diag(np.arange(N))
    return -A

if __name__ == "__main__":
    HiPPO = make_HiPPO(10)
    print('P')
    print(np.sqrt(1 + 2 * np.arange(10)))
    print('HiPPO')
    print(HiPPO.shape)
    print(HiPPO)

    SSMLayer = cloneLayer(SSMLayer)

    BatchStackedModel = nn.vmap(
    StackedModel,
    in_axes=0,
    out_axes=0,
    variable_axes={"params": None, "dropout": None, "cache": 0, "prime": None},
    split_rngs={"params": False, "dropout": True},)
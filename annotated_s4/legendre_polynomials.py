from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

"""
1. Creation of Coefficients Array (coef): For each basis function, an array coef of length N is created. This array is initially filled with zeros.

2. Setting One Coefficient: For the i-th basis function, the coef[N - i - 1] element is set to 1. This effectively selects the i-th Legendre polynomial.

3. Evaluating the Basis Function: The basis function f is then evaluated using numpy.polynomial.legendre.Legendre(coef)(t). This step computes the i-th Legendre polynomial over the domain t.
"""

def example_legendre(N=8):
    # Random hidden state as coefficients
    import numpy as np
    import numpy.polynomial.legendre

    # Each x[i](i=1..N) is a coefficient of a Legendre basis function. 
    # It scales the corresponding basis function, affecting its contribution to the overall function approximation.
    x = (np.random.rand(N) - 0.5) * 2
    # t is the domain, a range of input values over which the Legendre polynomials are evaluated.
    t = np.linspace(-1, 1, 100)
    f = numpy.polynomial.legendre.Legendre(x)(t)

    # Plot
    import matplotlib.pyplot as plt
    import seaborn

    seaborn.set_context("talk")
    #fig = plt.figure()
    #ax = fig.gca(projection="3d")
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(
        np.linspace(-25, (N - 1) * 100 + 25, 100),
        [0] * 100,
        zs=-1,
        zdir="x",
        color="black",
    )
    ax.plot(t, f, zs=N * 100, zdir="y", c="r")
    for i in range(N):
        coef = [0] * N
        coef[N - i - 1] = 1
        ax.set_zlim(-4, 4)
        ax.set_yticks([])
        ax.set_zticks([])
        # Plot basis function.
        f = numpy.polynomial.legendre.Legendre(coef)(t)
        ax.bar(
            [100 * i],
            [x[i]],
            zs=-1,
            zdir="x",
            label="x%d" % i,
            color="blue",
            fill=False,
            width=50,
        )
        ax.plot(t, f, zs=100 * i, zdir="y", c="green", alpha=0.5)
    ax.view_init(elev=40.0, azim=-45)
    fig.savefig("images/leg.png")

if __name__ == "__main__":
    example_legendre()
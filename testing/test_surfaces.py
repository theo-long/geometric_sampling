from manifold_sampling.surfaces import Torus, Sphere, SimpleAlgebraicIntersection, TriFold
import numpy as np

class SurfaceInstantiationTest():

    def torus_test():
        torus = Torus(r=1, R=1)
        assert np.allclose(torus.constraint_equation(1.0, 0.0, 0.0), 0.0)

    def sphere_test():
        sphere = Sphere(r=100, n_dim=5)
        assert np.allclose(sphere.constraint_equation(0.0, 10.0, 0.0, 0.0, 0.0), 0.0)

    def trifold_test():
        trifold = TriFold(power=3)
        assert np.allclose(trifold.constraint_equation(-1.0, -1.0, 2.0 ** (1/3)), 0.0)

    def algebraic_intersection_test():
        surface = SimpleAlgebraicIntersection()
        assert np.allclose(surface.constraint_equation(0.0, 0.0, 0.0), 0.0)
__doc__ = """"""

import numpy as np
from abc import ABC, abstractmethod
from elastica._linalg import _batch_matvec, _batch_cross
from elastica.utils import MaxDimension


class RigidBodyBase(ABC):
    """
    Base class for rigid body classes.

    Notes
    -----
    All rigid body class should inherit this base class.

    """

    def __init__(self):

        self.position_collection = NotImplementedError
        self.velocity_collection = NotImplementedError
        self.acceleration_collection = NotImplementedError
        self.omega_collection = NotImplementedError
        self.alpha_collection = NotImplementedError
        self.director_collection = NotImplementedError

        self.external_forces = NotImplementedError
        self.external_torques = NotImplementedError

        self.mass = NotImplementedError

        self.mass_second_moment_of_inertia = NotImplementedError
        self.inv_mass_second_moment_of_inertia = NotImplementedError

    # @abstractmethod
    #     # def update_accelerations(self):
    #     #     pass

    # def _compute_internal_forces_and_torques(self):
    #     """
    #     This function here is only for integrator to work properly. We do not need
    #     internal forces and torques at all.
    #     Parameters
    #     ----------
    #     time
    #
    #     Returns
    #     -------
    #
    #     """
    #     pass

    def update_accelerations(self, time):
        np.copyto(
            self.acceleration_collection,
            (self.external_forces) / self.mass,
        )

        # I apply common sub expression elimination here, as J w
        J_omega = _batch_matvec(
            self.mass_second_moment_of_inertia, self.omega_collection
        )

        # (J \omega_L ) x \omega_L
        lagrangian_transport = _batch_cross(J_omega, self.omega_collection)

        np.copyto(
            self.alpha_collection,
            _batch_matvec(
                self.inv_mass_second_moment_of_inertia,
                (lagrangian_transport + self.external_torques),
            ),
        )

    def zeroed_out_external_forces_and_torques(self, time):
        # Reset forces and torques
        self.external_forces *= 0.0
        self.external_torques *= 0.0

    def compute_position_center_of_mass(self):
        """
        Return positional center of mass
        """
        return self.position_collection[..., 0].copy()

    def compute_translational_energy(self):
        """
        Return translational energy
        """
        return (
            0.5
            * self.mass
            * np.dot(
                self.velocity_collection[..., -1], self.velocity_collection[..., -1]
            )
        )

    def compute_rotational_energy(self):
        """
        Return rotational energy
        """
        J_omega = np.einsum(
            "ijk,jk->ik", self.mass_second_moment_of_inertia, self.omega_collection
        )
        return 0.5 * np.einsum("ik,ik->k", self.omega_collection, J_omega).sum()

    def compute_position_point(self, point=np.array([0., 0., 0.]), index: int = None):
        """
        Compute position in world frame of point of specified in local frame of node
        """
        if index is None:
            # batched implementation for all nodes
            position = self.position_collection + _batch_matvec(self.director_collection, point)
        else:
            # only compute for one node
            position = self.position_collection[..., index] + np.dot(self.director_collection[..., index], point)
        return position

    def compute_velocity_point(self, point=np.array([0., 0., 0.]), index: int = None):
        """
        Compute velocity in world frame of point specified in local frame of node
        """
        if index is None:
            # batched implementation for all nodes
            velocity = self.velocity_collection + _batch_cross(self.omega_collection, point)
        else:
            # only compute for one node
            velocity = self.velocity_collection[..., index] + np.dot(self.omega_collection[..., index], point)
        return velocity

    # def apply_external_forces_at_point(self, force, point=np.array([0., 0., 0.]), index: int = None):
    #     """
    #     Apply force at point in local frame of node
    #     """
    #     if index is None:
    #         # batched implementation for all nodes
    #         self.external_forces += _batch_matvec(self.director_collection, force)
    #     else:
    #         # only compute for one node
    #         self.external_forces[..., index] += np.dot(self.director_collection[..., index], force)

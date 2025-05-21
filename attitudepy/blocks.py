"""Block functions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from dynamics import ABCDynamicsSimulator


class Block(ABC):  # noqa: D101
    @abstractmethod
    def __init__(self,
            guidance: Optional[Callable[[float, np.ndarray], np.ndarray]] = None,
            following: Block = None, sample_time: float = -1,
            custom_output: Optional[Callable] = None) -> None:
        """ABC initialiser.

        Parameters
        ----------
        guidance: callable
            Guidance law to be applied to calculate the state difference.
            The guidance law should take the time as first input and the state as second
        following: Block
            Block to be placed after this block (the output of this block
            is the input of the following block)
        custom_output: Callable
            Called insted of the default output method.
            It must have the same signatures as the control commands.
        """
        self.guidance = guidance
        self.following = following
        self.custom_output = custom_output
        self.sample_time = sample_time
        self.cached_u = None
        self.last_activation_time = -1
        self.cached_state = None

    def set_following(self, following: Block) -> None:
        """Set the following block.

        Parameters
        ----------
        following: Block
            Block to be palced after this block

        Raises
        ------
        RuntimeError
            If the following attribute is already set.
        """
        if self.following is not None:
            msg = "Following block already set."
            raise RuntimeError(msg)

        self.following = following

    def full_output(self, dynamics_simulator: ABCDynamicsSimulator, t: float,
                        block_input: Union[float, np.ndarray] = None,
                        ) -> Union[float, np.ndarray]:
        """Compute the full output given the state of the dynamics simulator.

        If there is a .following block, the output output will be the
        output of following receiving as input the output of this
        object.

        Parameters
        ----------
        dynamics_simulator: ABCDynamicsSimulator
            Dynamic simulator object
        t: float
            Current time
        block_input: float | ndarray
            Control command computed by the previous block.

        Returns
        -------
        float | ndarray
            Control command.
        """
        if self.following is None:
            return self.output(dynamics_simulator, t, block_input)
        return self.following.full_output(dynamics_simulator, t,
                                            self.output(dynamics_simulator, t,
                                                                block_input))

    def output(self, dynamics_simulator: ABCDynamicsSimulator, t: float,
                        block_input: Union[float, np.ndarray] = None,
                        ) -> Union[float, np.ndarray]:
        """Compute the output given the current state of dynamics simulator.

        Parameters
        ----------
        dynamics_simulator: ABCDynamicsSimulator
            Dynamic simulator object
        t: float
            Current time
        block_input: float | ndarray
            Control command computed by the previous block.

        Returns
        -------
        float | ndarray
            Control command.
        """
        if self.sample_time < 0:
            if self.custom_output is not None:
                return self.custom_output(dynamics_simulator, t,
                                                    block_input)
            return self._default_output(dynamics_simulator, t,
                                                    block_input)

        if (t - self.last_activation_time) >= self.sample_time:
            self.last_activation_time = t
            if self.custom_output is not None:
                self.cached_u = self.custom_output(dynamics_simulator, t,
                                                    block_input)
            self.cached_u = self._default_output(dynamics_simulator, t,
                                                    block_input)
        return self.cached_u

    @abstractmethod
    def _default_output(self,
                        dynamics_simulator: ABCDynamicsSimulator, t: float,
                        block_input: Union[float, np.ndarray] = None,
                        ) -> Union[float, np.ndarray]:
        """Set default output given the current state of dynamics simulator.

        Parameters
        ----------
        dynamics_simulator: ABCDynamicsSimulator
            Dynamic simulator object
        t: float
            Current time
        block_input: float | ndarray
            Control command computed by the previous block.

        Returns
        -------
        float | ndarray
            Control command.
        """
        return


class PIDController(Block):
    """PID controller.

    Attributes
    ----------
    kp: float | ndarray
        Proportional gains
    kd: float | ndarray
        Derivative gains
    guidance: callable
        Guidance law to be applied to calculate the state difference.
        The guidance law should take the time as first input and the state as second
    following: Block
        Block to be placed after this block (the output of this block
        is the input of the following block)
    """

    def __init__(self, kp: Union[float, np.ndarray], ki: Union[float, np.ndarray],
                    kd: Union[float, np.ndarray],
                    guidance: Callable[[float, np.ndarray], np.ndarray],
                    following: Block = None, sample_time: float = -1,
                    custom_output: Optional[Callable] = None) -> None:
        """Initialise the PD controller.

        Parameters
        ----------
        kp: float | ndarray
            Proportional gains
        ki: float | ndarray
            Inetgral gains. Works only for discrete control.
        kd: float | ndarray
            Derivative gains
        guidance: callable
            Guidance law to be applied to calculate the state difference.
            The guidance law should take the time as first input and the state as second
        following: Block
            Block to be placed after this block (the output of this block
            is the input of the following block)
        custom_output: Callable
            Called insted of the default output method.
            It must have the same signatures as the control commands.

        Raises
        ------
        ValueError
            If ki is not 0 and sample time is negative.
        """
        super().__init__(guidance, following, sample_time, custom_output)
        self.kp = kp
        self.kd = kd
        self.ki = ki
        if np.any(self.ki != 0) and self.sample_time < 0:
            msg = ("Integral control is not supported for continuous control. "
                            "Please set a sample time.")
            raise ValueError(msg)

        self.prev_input = None

        # avoid division by zero, for continous control derivative estimation
        # (not when calculated from attitude)
        self.last_t = -0.001

    def _default_output(self,
                        dynamics_simulator: ABCDynamicsSimulator, t: float,
                        block_input: Union[float, np.ndarray] = None,
                        ) -> Union[float, np.ndarray]:
        """Set default output given the current state of dynamics simulator.

        Parameters
        ----------
        dynamics_simulator: ABCDynamicsSimulator
            Dynamic simulator object
        t: float
            Current time
        block_input: float | ndarray
            Control command computed by the previous block.

        Returns
        -------
        ndarray
            Control command.
        """
        if self.sample_time < 0:
            return self._default_continuous_control_command(dynamics_simulator, t,
                                                    block_input)

        return self._default_discrete_control_command(dynamics_simulator, t,
                                                    block_input)

    def _default_continuous_control_command(self,
                        dynamics_simulator: ABCDynamicsSimulator, t: float,
                        block_input: Union[float, np.ndarray] = None,
                        ) -> Union[float, np.ndarray]:
        """Set default output given the current state of dynamics simulator.

        Parameters
        ----------
        dynamics_simulator: ABCDynamicsSimulator
            Dynamic simulator object
        t: float
            Current time
        block_input: float | ndarray
            Control command computed by the previous block.

        Returns
        -------
        ndarray
            Control command.
        """
        if block_input is None:
            sc = dynamics_simulator.spacecraft
            ref = self.guidance(t, sc.attitude.x)
            e, e_dot = sc.attitude.state_error(ref[:-3], ref[-3:], sc.mean_motion)
            if len(e) == 4:
                e_dot = np.append(e_dot, 0)

            e_int = self.compute_eint(e, t - self.last_t)
            self.prev_input = e

        else:
            e_dot = self.compute_edot(block_input, t - self.last_t)
            e_int = self.compute_eint(block_input, t - self.last_t)
            self.last_t = t
            self.prev_input = block_input

        return self.control_law(e, e_int, e_dot)

    def _default_discrete_control_command(self,
                        dynamics_simulator: ABCDynamicsSimulator, t: float,
                        block_input: Union[float, np.ndarray] = None,
                        ) -> Union[float, np.ndarray]:
        """Set default output given the current state of dynamics simulator.

        Parameters
        ----------
        dynamics_simulator: ABCDynamicsSimulator
            Dynamic simulator object
        t: float
            Current time
        block_input: float | ndarray
            Control command computed by the previous block.

        Returns
        -------
        ndarray
            Control command.
        """
        if block_input is None:
            sc = dynamics_simulator.spacecraft
            ref = self.guidance(t, sc.attitude.x)
            e, _ = sc.attitude.state_error(ref[:-3], ref[-3:], sc.mean_motion)

            e_int = self.compute_eint(e, self.sample_time)
            self.prev_input = e

            e_dot = sc.attitude.w if len(e) == 3 else np.append(sc.attitude.w, 0)

        else:
            e = block_input
            e_dot = self.compute_edot(block_input, self.sample_time)
            e_int = self.compute_eint(block_input, self.sample_time)

            self.prev_input = block_input

        return self.control_law(e, e_int, e_dot)

    def compute_edot(self, block_input: Union[float, np.ndarray],
                        time_step: float) -> Union[float, np.ndarray]:
        """Compute the input derivative.

        Returns
        -------
        float | ndarray
            Derivative of the input
        """
        if self.prev_input is None:
            self.prev_input = block_input

        return (block_input - self.prev_input) / time_step

    def compute_eint(self, block_input: Union[float, np.ndarray],
                        time_step: float) -> Union[float, np.ndarray]:
        """Compute the integrated input.

        Returns
        -------
        float | ndarray
            Integration of the input
        """
        if self.prev_input is None:
            self.prev_input = block_input

        return 0.5 * time_step * (block_input + self.prev_input)

    def control_law(self, e: Union[float, np.ndarray], e_int: Union[float, np.ndarray],
                e_dot: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute the control variable given input error and its derivative.

        Parameters
        ----------
        e: float | ndarray
            Input error
        e_dot: float | ndarray
            Input error derivative

        Returns
        -------
        float | ndarray
            Control command.
        """
        return -(e * self.kp + e_int * self.ki + e_dot * self.kd)


class NDIModelBased(Block):
    """Nonlinear Dynamic Inversion - model based approach.

    Requires a controller block, since it takes its output as input.
    The sample time is inherited from the previous controller.

    Attributes
    ----------
    following: Block
        Block to be placed after this block (the output of this block
        is the input of the following block)
    """

    def __init__(self, following: Block = None,
                    custom_output: Optional[Callable] = None) -> None:
        """Initialise the model based NDI block.

        Parameters
        ----------
        following: Block
            Block to be placed after this block (the output of this block
            is the input of the following block)
        custom_output: Callable
            Called insted of the default output method.
            It must have the same signatures as the control commands.
        """
        super().__init__(None, following, -1, custom_output)

    def _default_output(self,  # noqa: PLR6301
                        dynamics_simulator: ABCDynamicsSimulator, t: float,
                        block_input: np.ndarray,
                        ) -> np.ndarray:
        """Set default output given the current state of dynamics simulator.

        Parameters
        ----------
        dynamics_simulator: ABCDynamicsSimulator
            Dynamic simulator object
        t: float
            Current time.
        block_input: ndarray
            Control command computed by the previous block.

        Returns
        -------
        ndarray
            Control command.
        """
        ds = dynamics_simulator
        sc = ds.spacecraft
        anglen = len(sc.attitude.ang)
        # return ds.inv_gmatrix() @ (block_input[:3] - ds.fx())
        m_matrix = sc.attitude.nwdx_matrix @ np.vstack([np.zeros([anglen, 3]),
                                                        ds.gmatrix])

        w = sc.attitude.w if anglen == 3 else np.append(sc.attitude.w, 0)
        nw = sc.attitude.w2angdot_matrix() @ w
        l_vector = sc.attitude.nwdx_matrix @ np.append(nw.T, ds.fx.T)

        return np.linalg.inv(m_matrix[:3, :]) @ (block_input[:3] - l_vector[:3])


class NDITimeScaleSeparation(NDIModelBased):
    """Nonlinear Dynamic Inversion - time-scale separation approach.

    Requires a previous controller, since it takes its output as input

    Attributes
    ----------
    following: Block
        Block to be placed after this block (the output of this block
        is the input of the following block)
    """

    def __init__(self, following: Block = None,
                    custom_output: Optional[Callable] = None) -> None:
        """Initialise the timescale separation NDI block.

        The sample time is inherited from the previous controller.

        Parameters
        ----------
        following: Block
            Block to be placed after this block (the output of this block
            is the input of the following block)
        custom_output: Callable
            Called insted of the default output method.
            It must have the same signatures as the control commands.
        """
        super().__init__(following, custom_output)

    def _default_output(self,
                        dynamics_simulator: ABCDynamicsSimulator, t: float,
                        block_input: np.ndarray,
                        ) -> np.ndarray:
        """Set default output given the current state of dynamics simulator.

        Parameters
        ----------
        dynamics_simulator: ABCDynamicsSimulator
            Dynamic simulator object
        t: float
            Current time.
        block_input: ndarray
            Control command computed by the previous block.

        # Returns
        # -------
        # ndarray
        #     Control command.
        """
        # ds = dynamics_simulator
        # outer_loop_control = np.linalg.inv(ds.spacecraft.attitude.w2angdot_matrix()) \
        #                             @ block_input
        # return super()._default_output(ds, t, outer_loop_control)
        msg = "Not implemented yet"
        raise NotImplementedError(msg)


class ControlLoop(Block):
    """Defines a control loop."""

    def __init__(self, *controllers: Block) -> None:
        """Initialise the control loop.

        Parameters
        ----------
        controllers: Tuple[Block]
            Controllers forming the control loop.
        """
        self.controllers = list(controllers)

        for i in range(len(self.controllers) - 1):
            self.controllers[i].set_following(self.controllers[i + 1])

        # The first controller will handle output chaining
        self.entry_block = self.controllers[0]

        # Store attributes to fulfill Block interface
        self.guidance = self.entry_block.guidance
        self.sample_time = self.entry_block.sample_time
        self.custom_output = self.entry_block.custom_output
        self.following = None
        self.cached_u = None
        self.last_activation_time = -1
        self.cached_state = None

    def __str__(self) -> str:
        """Readable summary of the control loop."""  # noqa: DOC201
        block_names = " -> ".join(type(ctrl).__name__ for ctrl in self.controllers)
        return f"ControlLoop({block_names})"

    def __repr__(self) -> str:
        """Unambiguous representation for debugging."""  # noqa: DOC201
        return (f"<ControlLoop with {len(self.controllers)} blocks: "
                f"{' -> '.join(repr(ctrl) for ctrl in self.controllers)}>")

    def set_following(self, following: Block) -> None:
        """Set the following block.

        Parameters
        ----------
        following: Block
            Block to be palced after this block

        Raises
        ------
        RuntimeError
            If the following attribute is already set.
        """
        if self.following is not None:
            msg = "Following block already set."
            raise RuntimeError(msg)
        self.following = following
        # Attach to the last controller in the internal chain
        self.controllers[-1].set_following(following)

    def full_output(self, dynamics_simulator: ABCDynamicsSimulator, t: float,
                    block_input: Union[float, np.ndarray] = None,
                    ) -> Union[float, np.ndarray]:
        """Compute the full output given the state of the dynamics simulator.

        If there is a .following block, the output output will be the
        output of following receiving as input the output of this
        object.

        Parameters
        ----------
        dynamics_simulator: ABCDynamicsSimulator
            Dynamic simulator object
        t: float
            Current time
        block_input: float | ndarray
            Control command computed by the previous block.

        Returns
        -------
        float | ndarray
            Control command.
        """
        return self.entry_block.full_output(dynamics_simulator, t,
                                            block_input)

    def output(self, dynamics_simulator: ABCDynamicsSimulator, t: float,
                block_input: Union[float, np.ndarray] = None,
                ) -> Union[float, np.ndarray]:
        """Compute the output given the current state of dynamics simulator.

        Parameters
        ----------
        dynamics_simulator: ABCDynamicsSimulator
            Dynamic simulator object
        t: float
            Current time
        block_input: float | ndarray
            Control command computed by the previous block.

        Returns
        -------
        float | ndarray
            Control command.
        """
        return self.entry_block.output(dynamics_simulator, t, block_input)

    def _default_output(self, dynamics_simulator: ABCDynamicsSimulator, t: float,
                        block_input: Union[float, np.ndarray] = None,
                        ) -> Union[float, np.ndarray]:
        # Delegate to the first block
        return self.entry_block._default_output(dynamics_simulator, t, block_input)  # noqa: SLF001


class ClassicNDIControlLoop(ControlLoop):
    """Defines the NDI control loop."""

    def __init__(self, controller: PIDController,
                    ndi_loop: NDIModelBased) -> None:
        """Initialise the NDI control loop.

        Parameters
        ----------
        controller: PIDController
            Controller of the NDI loop
        ndi_loop: NDIModelBased
            Model based NDI loop
        """
        super().__init__(controller, ndi_loop)

import torch as th

from adarl.utils.dbg.dbg_checks import dbg_check
from adarl.utils.utils import vector_projection


def set_column(t : th.Tensor, idx :th.Tensor, value : th.Tensor) -> th.Tensor:
    """ Set a column of a 2D tensor. Equivalent to t[:, idx] = value, but more friendly to th.compile.
    """
    t.T.index_put_([idx], value)
    return t


def planar_tracking_error_vec(body_rel_linvel_vec_xyz : th.Tensor, gravity_rel_vec_xyz : th.Tensor, goal_rel_linvel_vec_xyz : th.Tensor) -> th.Tensor:
        """_summary_

        Parameters
        ----------
        body_rel_linvel_vec_xyz : th.Tensor
            current linvel, relative to the body frame
        gravity_rel_vec_xyz : th.Tensor
            gravity vector, relative to the body frame
        goal_rel_linvel_vec_xyz : th.Tensor
            linvel goal, relative to the body frame


        Returns
        -------
        th.Tensor
            _description_
        """
        body_planar_rel_linvel_xyz = body_rel_linvel_vec_xyz - vector_projection(body_rel_linvel_vec_xyz,gravity_rel_vec_xyz)
        # goal_rel_linvel_xyz should already be "planar", it's projection along gravity_rel should be zero
        eps = 1e-3
        norms = th.norm(vector_projection(goal_rel_linvel_vec_xyz,gravity_rel_vec_xyz), dim = 1)
        dbg_check(lambda: th.all(norms < eps) == True,
                  lambda:   f"goal_rel_linvel_xyz is not horizontal (th.all(norms < {eps}) = {th.all(norms < eps)}), projection is "
                            f"{vector_projection(goal_rel_linvel_vec_xyz, gravity_rel_vec_xyz)[th.logical_or(norms >= eps,th.logical_not(th.isfinite(norms)))]}"
                            f"goal={goal_rel_linvel_vec_xyz[th.logical_or(norms >= eps,th.logical_not(th.isfinite(norms)))]}"
                            f"gravity={gravity_rel_vec_xyz[th.logical_or(norms >= eps,th.logical_not(th.isfinite(norms)))]}"
                            f" big={th.nonzero(norms >= eps)}"
                            f" isnan={th.nonzero(th.isnan(norms))}"
                            f" isinf={th.nonzero(th.isinf(norms))}",
                    async_assert=True,
                    assert_msg="goal_rel_linvel_vec_xyz is not horizontal")
        return th.linalg.norm(body_planar_rel_linvel_xyz-goal_rel_linvel_vec_xyz, dim = 1)


def smooth_clip(x : th.Tensor, max_value : float, softness : float) -> th.Tensor:
    """ Smoothly clips x to be under max_value, with a softness parameter.

    Parameters
    ----------
    x : th.Tensor
        Input tensor
    max_value : th.Tensor
        Maximum value
    softness : th.Tensor
        Softness parameter, lower values result in a softer clipping

    Returns
    -------
    th.Tensor
        Clipped tensor
    """
    return x/(1+th.abs(x/max_value)**softness)**(1/softness)


def norm_penalty_flat2(x : th.Tensor,
                 norm : float,
                 power : float,
                 squash_max : float = 1.0,
                 squash_smoothness : float = 4.0,
                 flattening_threshold : float = 0.0,
                 flattening_width : float = 2.0,
                 scale : float = 1.0) -> th.Tensor:
    """ A penalty applied to the norm of the vectors in x. The penalty is computed as the norm raised to the power of 'power',
        and then flattened using a smooth clipping function.

    Parameters
    ----------
    x : th.Tensor
        Input tensor (batch_size, n_dimensions)
    norm : float
        Norm to use for the penalty
    power : float
        Power to raise the norm to
    max_val : float
        Maximum value for the penalty
    squash_smoothness : float
        Smoothness parameter for the clipping
    flattening_threshold : float
        Threshold at which flattening starts
    flattening_width : float
        Width of the flattening ramp (the flattening reaches 1 at threshold + width)
    scaling : float
        Output will be scaled to be maximum this value

    Returns
    -------
    th.Tensor
        Penalty tensor
    """
    joint_norms = th.linalg.norm(x, dim=1, ord=norm)
    base_penalty = th.pow(joint_norms, power)
    squashed_penalties = smooth_clip(base_penalty, squash_max, squash_smoothness)
    if flattening_width > 0.0:
         # a smooth ramp that starts around flattening_threshold and reaches 1 around flattening_threshold+flattening_width
        t = flattening_threshold
        w = flattening_width
        flattening = smooth_clip(((joint_norms-t)*2-1)/w-1, 1.0, 10.0)+1
        penalties = squashed_penalties*flattening
    else:
        penalties = squashed_penalties
    penalties = penalties/squash_max * scale
    return -penalties


def smoothclip_flattener(x : th.Tensor, t : float, w : float) -> th.Tensor:
    """ Flattening function based on smoothclip. Use it as f(x)*smoothclip_flattener(x) = flattened_f(x).

    Parameters
    ----------
    x : th.Tensor
        _description_
    t : float
        threshold below which flattening starts
    w : float
        width of the flattening ramp

    Returns
    -------
    th.Tensor
        flattening coefficient
    """
    return (smooth_clip(((x-t)*2-1)/w-1, 1.0, 8.0)+1)/2


def flattener(x: th.Tensor, scale: float, power : float) -> th.Tensor:
    """ Flattens the input tensor x using a smooth clipping function. The flattening is done by multiplying x with (1 - exp(-(x**power/scale))).

    Parameters
    ----------
    x : th.Tensor
        Input tensor
    scale : float
        Scale parameter for the flattening

    Returns
    -------
    th.Tensor
        Flattened tensor
    """
    return x * (1 - th.exp(-(x**power/scale)))


def norm_penalty(x : th.Tensor,
                 norm : float,
                 power : float,
                 squash_max : float = 1.0,
                 squash_smoothness : float = 4.0,
                 flattening_scale : float = 0.0,
                 flattening_power : float = 2.0,
                 scale : float | None = None) -> th.Tensor:
    """ A penalty applied to the norm of the vectors in x. The penalty is computed as the norm raised to the power of 'power',
        and then flattened using a smooth clipping function.

    Parameters
    ----------
    x : th.Tensor
        Input tensor (batch_size, n_dimensions)
    norm : float
        Norm to use for the penalty
    power : float
        Power to raise the norm to
    squash_max : float
        Maximum value for the penalty
    squash_smoothness : float
        Smoothness parameter for the clipping (lower is smoother, see smooth_clip)
    flattening_scale : float
        Scale parameter for the flattening, see flattener()
    flattening_power : float
        Power parameter for the flattening, see flattener()
    scale : float | None
        Output will be scaled to be maximum this value, if not None. If None, no scaling is applied.

    Returns
    -------
    th.Tensor
        Penalty tensor
    """
    joint_norms = th.linalg.norm(x, dim=1, ord=norm)
    base_penalty = th.pow(joint_norms, power)
    squashed_penalties = smooth_clip(base_penalty, squash_max, squash_smoothness)
    if flattening_scale > 0.0:
        flattened_penalties = flattener(squashed_penalties, flattening_scale, flattening_power)
        penalties = flattened_penalties
    else:
        penalties = squashed_penalties
    if scale is not None:
        penalties = penalties/squash_max * scale
    return -penalties


def flattened_penalty_reward(x, max_rew : float, exponent : float, flattening_scale : float):
    """A penalty produced by raising abs(x) at the power of exponent, and flattening it with
        a flipped exponential, scaled with flattening_scale. With exponent=1.5 and 
        flattening_scale=0.1 results in an x^1.5 that is quite flat below 0.1.
        This then is squashed with a tanh to be under max_rew.
        In formulas (not squashed): x^exponent * (-e^(-x^2/flattening_scale)+1)
    """
    return -th.tanh((th.pow(th.abs(x),exponent)*(1-th.exp(-(x/flattening_scale)**2)))/max_rew)*max_rew


def penalty_reward(x, max_rew : float, exponent : float):
    """A penalty produced by raising abs(x) at the power of exponent, and squashing
        it with a tanh to be under max_rew.
    """
    return -th.tanh(th.pow(th.abs(x),exponent)/max_rew)*max_rew


def joint_penalty_reward(x, max_rew : float, exponent : float, reduction : str = "mean", presquash_factor : float = 1.0):
    """A penalty produced by raising abs(x) at the power of exponent, and squashing
        it with a tanh to be under max_rew.
    """
    if reduction == "mean":
        return -th.tanh(th.mean(th.pow(th.abs(x),exponent),dim=1)*presquash_factor/max_rew)*max_rew
    elif reduction == "sum":
        return -th.tanh(th.sum( th.pow(th.abs(x),exponent),dim=1)*presquash_factor/max_rew)*max_rew
    elif reduction == "max":
        return -th.tanh(th.amax(th.pow(th.abs(x),exponent),dim=1)*presquash_factor/max_rew)*max_rew
    else:
        raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction}")


def flattened_joint_penalty_reward(x, max_rew : float, exponent : float, flattening_scale : float, presquash_factor : float = 1.0):
    """A penalty produced by raising abs(x) at the power of exponent, and flattening it with
        a flipped exponential, scaled with flattening_scale. With exponent=1.5 and 
        flattening_scale=0.1 results in an x^1.5 that is quite flat below 0.1.
        This then is squashed with a tanh to be under max_rew.
        In formulas (not squashed): x^exponent * (-e^(-x^2/flattening_scale)+1)
    """
    return -th.tanh((th.mean(th.pow(th.abs(x),exponent)*(1-th.exp(-(x/flattening_scale)**2)), dim=1))*presquash_factor/max_rew)*max_rew


def ramp_reward(error : th.Tensor, zero_rew_dist : th.Tensor):
    return 1-error/zero_rew_dist


def bell_reward(error : th.Tensor, zero_rew_dist : th.Tensor | float):
    """A bell-shaped reward function. It's 1 at error = 0, it reaches about zero (~0.0183) at error = zero_rew_dist

    Parameters
    ----------
    error : th.Tensor
        Error value
    zero_rew_dist : th.Tensor | float
        Error value at which the reward should start to settle around zero

    Returns
    -------
    th.Tensor
        Reward value
    """
    return th.exp(-(2*error/zero_rew_dist)**2)


def double_bell_reward(error : th.Tensor, bell_width_a : th.Tensor, bell_width_b : th.Tensor, bell_b_weight : th.Tensor):
    return (   bell_b_weight  * bell_reward(error, zero_rew_dist=bell_width_b)+
            (1-bell_b_weight) * bell_reward(error, zero_rew_dist=bell_width_a))
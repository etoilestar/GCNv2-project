import numpy as np


def validate_number_sequence(seq, n):
    """Validate a sequence to be of a certain length and ensure it's a numpy array of floats.
    Raises:
        ValueError: Invalid length or non-numeric value
    """
    if seq is None:
        return np.zeros(n)
    if len(seq) == n:
        try:
            l = [float(e) for e in seq]
        except ValueError:
            raise ValueError(
                "One or more elements in sequence <{!r}> cannot be interpreted as a real number".format(seq))
        else:
            return np.asarray(l)
    elif len(seq) == 0:
        return np.zeros(n)
    else:
        raise ValueError("Unexpected number of elements in sequence. Got: {}, Expected: {}.".format(len(seq), n))

def slerp(q0, q1, amount=0.5):
    """
    Params:
        q0: first endpoint rotation
        q1: second endpoint rotation
        amount: interpolation parameter between 0 and 1. This describes the linear placement position of
            the result along the arc between endpoints; 0 being at `q0` and 1 being at `q1`.
            Defaults to the midpoint (0.5).
    Returns:
        A new endpoint
    """
    # # Ensure quaternion inputs are unit quaternions and 0 <= amount <=1
    # q0._fast_normalise()
    # q1._fast_normalise()
    amount = np.clip(amount, 0, 1)

    dot = np.dot(q0, q1)

    # If the dot product is negative, slerp won't take the shorter path.
    # Note that v1 and -v1 are equivalent when the negation is applied to all four components.
    # Fix by reversing one quaternion
    if dot < 0.0:
        q0 = -q0
        dot = -dot

    # sin_theta_0 can not be zero
    if dot > 0.9995:
        qr = validate_number_sequence(q0 + amount * (q1 - q0), 4)
        return qr

    theta_0 = np.arccos(dot)  # Since dot is in range [0, 0.9995], np.arccos() is safe
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * amount
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    qr = validate_number_sequence((s0 * q0) + (s1 * q1), 4)
    return qr

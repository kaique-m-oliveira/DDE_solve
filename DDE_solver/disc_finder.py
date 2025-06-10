import numpy as np
from scipy.optimize import root


def find_discontinuity_chain(
    alpha_func, t0_initial, t_end, h_guess, max_iterations=100
):
    """
    Locates the discontinuity points for non vanishing time delays
    """
    discs = [t0_initial]
    current_tk = t0_initial

    for _ in range(max_iterations):
        if current_tk >= t_end:
            break  # Stop if beyond end time

        # Function to find root: G(t) = alpha(t) - current_tk = 0
        func_for_root = lambda t_val: alpha_func(t_val) - current_tk

        # Initial guess for the next root. Must be > current_tk.
        guess = current_tk + h_guess
        if guess <= current_tk:
            guess = current_tk + 1e-6  # Ensure strictly greater

        try:
            result = root(func_for_root, guess)

            if result.success:
                next_tk = result.x[0]  # Extract scalar root

                # Check constraints: t_k < t_{k+1} and t_{k+1} <= t_end
                if next_tk > current_tk + 1e-9 and next_tk <= t_end + 1e-9:
                    discs.append(next_tk)
                    current_tk = next_tk
                else:
                    break  # Root found is not in desired sequence or range
            else:
                break  # Root finding failed for this iteration
        except Exception:
            break  # Catch potential errors during root finding

    return discs

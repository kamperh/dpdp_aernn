"""
Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2021
"""

import numpy as np


def custom_viterbi(costs, n_frames):
    """
    Viterbi segmentation of an utterance of length `n_frames` based on `costs`.

    Parameters
    ----------
    costs : n_frames*(n_frames + 1)/2 array
        For t = 1, 2, ..., N the entries costs[i:i + t] contains the costs of
        seq[0:t] up to seq[t - 1:t], with i = t(t - 1)/2. Written out: costs =
        [cost(seq[0:1]), cost(seq[0:2]), cost(seq[1:2]), cost(seq[0:3]), ...,
        cost(seq[N-1:N])].

    Return
    ------
    (summed_cost, boundaries) : (float, array of bool)
    """
    
    # Initialise
    boundaries = np.zeros(n_frames, dtype=bool)
    boundaries[-1] = True
    alphas = np.ones(n_frames)
    alphas[0] = 0.0

    # Forward filtering
    i = 0
    for t in range(1, n_frames):
        alphas[t] = np.min(
            costs[i:i + t] + alphas[:t]
            )
        i += t

    # Backward segmentation
    t = n_frames
    summed_cost = 0.0
    while True:
        i = int(0.5*(t - 1)*t)
        q_t_min_list = (
            costs[i:i + t] + alphas[:t]       
            )
        q_t_min_list = q_t_min_list[::-1]
        q_t = np.argmin(q_t_min_list) + 1

        summed_cost += costs[i + t - q_t]
        if t - q_t - 1 < 0:
            break
        boundaries[t - q_t - 1] = True
        t = t - q_t

    return summed_cost, boundaries

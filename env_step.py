import numpy as np


def env_step(P, signal_power, noise_power, ch_bandwidth):
    # Each row is for AP, column for device
    # Uplink communication (according to figure)

    if P.size == 1:
        SINR = P / noise_power
        rate = ch_bandwidth * np.log2(1 + SINR)
    else:
        SINR = P.diagonal() / ((P.sum(axis=1) - P.diagonal()) + noise_power)
        rate = ch_bandwidth * np.log2(1 + SINR)

    SNR_max = signal_power.diagonal() / noise_power
    max_rate = ch_bandwidth * np.log2(1 + SNR_max)

    return rate, max_rate
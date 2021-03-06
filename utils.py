"""
Ben Hayes 2020

ECS7013P Deep Learning for Audio & Music

File: utils.py
Description: A handful of convenience definitions. These are hangovers from
             having originally written this code in a Jupyter notebook and
             should cease to exist after a refactor.
"""

SAMPLE_RATE = 16000
SAMPLE_LENGTH_IN_SECONDS = 4

param_ranges = {
    "f0": {"min": 20.0, "max": 0.25 * SAMPLE_RATE},
    "amplitude": {"min": 0.0, "max": 1.0}
}


def scale_param(param, range_name):
    return param\
        * (param_ranges[range_name]["max"]
            - param_ranges[range_name]["min"])\
        + param_ranges[range_name]["min"]

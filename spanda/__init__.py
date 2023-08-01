_param_defaults_s2p = {}

_param_defaults_dFoF = {
    "dFoF_calculation": {
        "neuropil_fraction": 0.7,
        "percentile_baseline": 30,
        "rolling_percentile_window": 18000,
        "roll_centered": False,
        "roll_stride": 10,
        "roll_interpolation": "nearest",
        "channelOffset_correction": 0,
        "multicore_pref": True,
        "verbose": True,
    },
    "tqm": {
        "var_ratio__Fneu_over_F": (0, 0.5),
        "EV__F_by_Fneu": (0, 0.7),
        "base_FneuSub": (100, 2000),
        "base_F": (200, 3500),
        "nsr_autoregressive": (0, 6),
        "noise_derivMAD": (0, 0.015),
        "max_dFoF": (0.75, 10),
        "baseline_var": (0, 0.015),
    },
}

_param_defaults_drifter = {
    "netrep_alpha": 1.0,
}

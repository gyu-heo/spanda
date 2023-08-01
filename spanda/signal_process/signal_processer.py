import os
from pathlib import Path
import itertools
import multiprocessing as mp

import numpy as np
import natsort
from spanda.spanda.util import file_helpers
import torch

import json
import pickle

from tqdm import tqdm

from spanda import _param_defaults_dFoF
from spanda.util import util
from spanda.spanda.signal_process.ca2p_preprocessing import (
    make_dFoF,
    trace_quality_metrics,
)


class dFoF_processer:
    def __init__(
        self,
        params_input=None,
        memory_safe: bool = False,
    ):
        """
        Pipeline class to create dFoF.

        Args:
            params_input (dict or str, optional):
                Overwrite parameters given inputs. Defaults to None.
                If None, use default parameters.
                If dict, overwrite default parameters' matching key-value pair with given dict.
                If str, load parameters from given json file.
        """
        self.params = _param_defaults_dFoF
        self.params_input = params_input
        self.memory_safe = memory_safe

        if params_input is not None:
            self.params = util.overwrite_params(self.params_input, self.params)

    def make_dFoFs(
        self,
        F_dict: dict,
        Fneu_dict: dict,
    ):
        """Make dFoF from F and Fneu. If memory_safe, iterate over each session.

        Args:
            F_dict (dict):
                keys: session
                values: F (np.ndarray)
            Fneu_dict (dict):
                keys: session
                values: Fneu (np.ndarray)
        """
        if self.memory_safe:
            for session in F_dict:
                out = make_dFoF(
                    F=F_dict[session],
                    Fneu=Fneu_dict[session],
                    **self.params["dFoF_calculation"],
                )
            # TODO save each array to file. Files should be nested: mouse - analysis_version - session - array
            # TODO analysis_version should contain all parameters used to create the dFoF
        else:
            outs = make_dFoF(
                F=F_dict,
                Fneu=Fneu_dict,
                **self.params["dFoF_calculation"],
            )

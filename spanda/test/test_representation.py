import unittest
import numpy as np

from spanda import _param_defaults_drifter
from spanda import util
from spanda.bnpm.timeseries import event_triggered_traces
from spanda.representation import drifter


class Test_Drifter(unittest.TestCase):
    """Test class for Drifter representation"""

    @classmethod
    def setUpClass(cls):
        """
        Here we create mock data and arguments for testing purposes.
        """
        ## Fake data: (num.of.samples, num.of.features) shape matrix
        cls.n_samples = 18000
        cls.n_features = 100
        cls.session_names = ["20230101", "20230102", "20230105", "20230106"]
        cls.test_matrices = {
            session: np.random.rand(cls.n_samples, cls.n_features)
            for session in cls.session_names
        }

        ## Create falsified sampling events
        cls.sampling_events = ["trial_onset", "threshold_crossing"]
        cls.sampling_indices = {
            session: {
                sampling_event: np.sort(np.random.randint(101, 17899, 50))
                for sampling_event in cls.sampling_events
            }
            for session in cls.session_names
        }
        cls.sampling_windows = {
            sampling_event: [-np.random.randint(0, 100), np.random.randint(0, 100)]
            for sampling_event in cls.sampling_events
        }

        cls.sampling_axis = 0

    def setUp(self):
        pass

    def test_drifter_default_params(self):
        """
        Test if default parameters in __init__.py work correctly
        """
        drifter_instance = drifter.drifter()
        self.assertEqual(drifter_instance.params, _param_defaults_drifter)

    def test_drifter_overwrite_params_dict(self):
        """
        Test if overwriting parameters in __init__.py work correctly
        """
        drifter_instance = drifter.drifter(
            params_input={
                "netrep_alpha": 0.5,
            }
        )
        self.assertEqual(drifter_instance.params["netrep_alpha"], 0.5)

    def test_drifter_overwrite_params_json(self):
        pass

    def test_drifter_create_ref_matrices(self):
        """
        Create reference matrices from mock data and arguments
        Test if the output has the correct shape
        """
        drifter_instance = drifter.drifter()
        ref_matrices = drifter_instance.create_ref_matrices(
            input_data=self.test_matrices,
            sampling_events=self.sampling_events,
            sampling_indices=self.sampling_indices,
            sampling_windows=self.sampling_windows,
            sampling_axis=self.sampling_axis,
        )
        self.assertEqual(len(ref_matrices), len(self.session_names))
        self.assertEqual(ref_matrices[0].shape[self.sampling_axis - 1], self.n_features)
        window_size = 0
        for sampling_event in self.sampling_events:
            window_size += np.diff(self.sampling_windows[sampling_event])
        self.assertEqual(ref_matrices[0].shape[self.sampling_axis], window_size)

    def test_drifter_fit(self):
        """
        Test if the fit method works correctly
        """
        pass

    def test_drifter_pairwise_distance(self):
        """
        Test if the pairwise distance method works correctly
        """
        pass


if __name__ == "__main__":
    unittest.main()

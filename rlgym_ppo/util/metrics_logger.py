from abc import ABC
from rlgym_sim.utils.gamestates import GameState
import numpy as np

class MetricsLogger(ABC):
    def collect_metrics(self, game_state: GameState) -> np.ndarray:
        metrics_arrays = self._collect_metrics(game_state)
        unraveled = []
        for arr in metrics_arrays:
            shape = np.shape(arr)
            unraveled.append(len(shape))
            unraveled += shape
            unraveled += np.ravel(arr).tolist()

        return np.asarray(unraveled).astype(np.float32)

    def report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        if wandb_run is None:
            return

        all_reports = []
        for serialized_metrics in collected_metrics:
            metrics_arrays = []
            i = 0
            while i < len(serialized_metrics):
                n_shape = int(serialized_metrics[i])
                n_values_in_metric = 1
                shape = []
                i += 1
                for arg in serialized_metrics[i:i+n_shape]:
                    n_values_in_metric *= arg
                    shape.append(int(arg))
                n_values_in_metric = int(n_values_in_metric)
                metric = serialized_metrics[i+n_shape:i+n_shape+n_values_in_metric]
                metrics_arrays.append(metric)
                i = i+n_shape+n_values_in_metric
            all_reports.append(metrics_arrays)

        self._report_metrics(all_reports, wandb_run, cumulative_timesteps)

    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        raise NotImplementedError

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        raise NotImplementedError
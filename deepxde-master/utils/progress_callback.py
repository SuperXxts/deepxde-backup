import time
import numpy as np
import deepxde as dde
import shutil

from tqdm.auto import tqdm


class TqdmProgressCallback(dde.callbacks.Callback):
    def __init__(
        self,
        total_steps,
        display_every=1000,
        true_solution_fn=None,
        metric_name="metric",
        ncols=100,
        bar_width=25,
        postfix_max_len=60,
        initial_step=0,
        step_offset=0,
    ):
        super().__init__()
        self.total_steps = int(total_steps) if total_steps is not None else 0
        self.display_every = max(int(display_every), 1)
        self.true_solution_fn = true_solution_fn
        self.metric_name = str(metric_name)
        self.ncols = int(ncols)
        self.bar_width = int(bar_width)
        self.postfix_max_len = int(postfix_max_len)
        self.initial_step = int(initial_step)
        self.step_offset = int(step_offset)

        self._bar = None
        self._last_step = int(initial_step)
        self._t0 = None
        self._last_metric_step = -1
        self._last_metric_value = None

    def on_train_begin(self):
        self._t0 = time.time()
        self._last_step = int(self.initial_step)
        self._last_metric_step = -1
        self._last_metric_value = None

        term_width = shutil.get_terminal_size(fallback=(120, 24)).columns
        effective_ncols = min(self.ncols, term_width)

        self._bar = tqdm(
            total=self.total_steps,
            initial=min(int(self.initial_step), self.total_steps),
            dynamic_ncols=False,
            ncols=effective_ncols,
            leave=True,
            bar_format=f"Epoch {{n_fmt}}/{{total_fmt}} |{{bar:{self.bar_width}}}| {{elapsed}}<{{remaining}} {{postfix}}",
        )

    def on_train_end(self):
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def _loss_scalar(self, loss_value):
        if loss_value is None:
            return None
        try:
            return float(np.sum(loss_value))
        except Exception:
            return None

    def _compute_metric(self):
        if self.true_solution_fn is None:
            return None

        X_test = getattr(self.model.train_state, "X_test", None)
        if X_test is None:
            return None

        x_for_true = X_test[0] if isinstance(X_test, (list, tuple)) else X_test
        try:
            y_true = self.true_solution_fn(np.asarray(x_for_true))
        except Exception:
            return None

        try:
            y_pred = self.model.predict(X_test)
        except Exception:
            return None

        y_pred_arr = np.asarray(y_pred[0] if isinstance(y_pred, (list, tuple)) else y_pred)
        y_true_arr = np.asarray(y_true)
        if y_true_arr.shape != y_pred_arr.shape:
            return None

        denom = np.linalg.norm(y_true_arr.ravel())
        if denom <= 1e-12:
            return 0.0
        return float(np.linalg.norm((y_true_arr - y_pred_arr).ravel()) / denom)

    def _build_postfix(self, train_loss, test_loss, metric_value):
        parts = [
            f"train={train_loss:.2e}" if train_loss is not None else "train=--",
            f"test={test_loss:.2e}" if test_loss is not None else "test=--",
            f"{self.metric_name}={metric_value:.2e}" if metric_value is not None else f"{self.metric_name}=--",
        ]
        s = " ".join(parts)
        if self.postfix_max_len > 0 and len(s) > self.postfix_max_len:
            s = s[: self.postfix_max_len]
        return s

    def on_batch_end(self):
        raw_step = int(getattr(self.model.train_state, "step", 0))
        step = raw_step + self.step_offset
        train_loss = self._loss_scalar(getattr(self.model.train_state, "loss_train", None))
        test_loss = self._loss_scalar(getattr(self.model.train_state, "loss_test", None))

        if step % self.display_every == 0 or step >= self.total_steps:
            metric = self._compute_metric()
            if metric is not None:
                self._last_metric_value = metric
                self._last_metric_step = step

        inc = step - self._last_step
        if inc > 0:
            self._bar.update(inc)
            self._last_step = step

        postfix_str = self._build_postfix(train_loss, test_loss, self._last_metric_value)
        self._bar.set_postfix_str(postfix_str, refresh=False)
        self._bar.refresh()

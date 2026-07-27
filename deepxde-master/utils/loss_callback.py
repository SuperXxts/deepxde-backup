import json
import os
from types import SimpleNamespace

import deepxde as dde

from .save_results import (
    plot_all_loss_components,
    plot_and_save_loss_history,
    plot_parameter_history,
    save_best_test_loss_json,
    save_loss_history_json,
)


def _load_history_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return {
        "steps": [int(step) for step in payload.get("steps", [])],
        "loss_train": payload.get("loss_train", []) or [],
        "loss_test": payload.get("loss_test", []) or [],
    }


def _merge_history_payloads(existing, current):
    merged_steps = []
    merged_train = []
    merged_test = []

    def add_row(step, train_row, test_row):
        step = int(step)
        train_row = list(train_row) if train_row is not None else []
        test_row = list(test_row) if test_row is not None else []
        while merged_steps and merged_steps[-1] > step:
            merged_steps.pop()
            merged_train.pop()
            merged_test.pop()
        if merged_steps and merged_steps[-1] == step:
            merged_train[-1] = train_row
            merged_test[-1] = test_row
            return
        merged_steps.append(step)
        merged_train.append(train_row)
        merged_test.append(test_row)

    existing_steps = existing.get("steps", []) if existing else []
    existing_train = existing.get("loss_train", []) if existing else []
    existing_test = existing.get("loss_test", []) if existing else []
    for index, step in enumerate(existing_steps):
        add_row(
            step,
            existing_train[index] if index < len(existing_train) else [],
            existing_test[index] if index < len(existing_test) else [],
        )

    current_steps = list(getattr(current, "steps", []) or [])
    current_train = list(getattr(current, "loss_train", []) or [])
    current_test = list(getattr(current, "loss_test", []) or [])
    for index, step in enumerate(current_steps):
        add_row(
            step,
            current_train[index] if index < len(current_train) else [],
            current_test[index] if index < len(current_test) else [],
        )

    return {
        "steps": merged_steps,
        "loss_train": merged_train,
        "loss_test": merged_test,
    }


def _history_payload_to_namespace(payload):
    return SimpleNamespace(
        steps=payload.get("steps", []),
        loss_train=payload.get("loss_train", []),
        loss_test=payload.get("loss_test", []),
    )


class ParameterPlottingCallback(dde.callbacks.Callback):
    def __init__(self, param_file, save_path, true_values=None, param_names=None, period=1000):
        super().__init__()
        self.param_file = param_file
        self.save_path = save_path
        self.true_values = true_values
        self.param_names = param_names
        self.period = period
        self.last_saved_step = -1

    def on_batch_end(self):
        current_step = self.model.train_state.step
        if current_step % self.period == 0 and current_step != self.last_saved_step:
            if os.path.exists(self.param_file):
                try:
                    plot_parameter_history(
                        self.param_file,
                        self.true_values,
                        self.param_names,
                        self.save_path,
                    )
                except Exception:
                    pass
            self.last_saved_step = current_step


class LossHistoryCallback(dde.callbacks.Callback):
    def __init__(
        self,
        save_dir,
        period=None,
        filename="损失历史详细图.png",
        num_pde_losses=None,
        num_bc_losses=None,
        pde_loss_names=None,
        bc_loss_names=None,
        pde_label="Physics Loss",
        bc_label="Boundary Loss",
        data_loss_prefix="obs_",
        data_label="Observation Loss",
        show_total=False,
        save_all_components=True,
        append_existing=False,
    ):
        super().__init__()
        self.save_dir = save_dir
        self.period = period
        self.filename = filename
        self.num_pde_losses = num_pde_losses
        self.num_bc_losses = num_bc_losses
        self.pde_loss_names = pde_loss_names
        self.bc_loss_names = bc_loss_names
        self.pde_label = pde_label
        self.bc_label = bc_label
        self.data_loss_prefix = data_loss_prefix
        self.data_label = data_label
        self.show_total = bool(show_total)
        self.save_all_components = save_all_components
        self.append_existing = bool(append_existing)
        self.last_saved_step = -1
        self._existing_history = None

    def on_train_begin(self):
        os.makedirs(self.save_dir, exist_ok=True)
        if self.append_existing:
            history_path = os.path.join(self.save_dir, "json", "loss_history.json")
            self._existing_history = _load_history_json(history_path)
        if self.period is None:
            self.period = 1000
        self._save_loss_history()

    def on_batch_end(self):
        current_step = self.model.train_state.step
        if current_step % self.period == 0 and current_step != self.last_saved_step:
            self._save_loss_history()
            self.last_saved_step = current_step

    def on_train_end(self):
        self._save_loss_history()

    def _build_effective_history(self):
        if not self.append_existing or not self._existing_history:
            return self.model.losshistory
        merged_payload = _merge_history_payloads(self._existing_history, self.model.losshistory)
        return _history_payload_to_namespace(merged_payload)

    def _save_loss_history(self):
        try:
            effective_history = self._build_effective_history()
            plot_and_save_loss_history(
                effective_history,
                self.save_dir,
                filename=self.filename,
                num_pde_losses=self.num_pde_losses,
                num_bc_losses=self.num_bc_losses,
                pde_label=self.pde_label,
                bc_label=self.bc_label,
                bc_loss_names=self.bc_loss_names,
                data_loss_prefix=self.data_loss_prefix,
                data_label=self.data_label,
                show_total=self.show_total,
            )
            save_loss_history_json(effective_history, self.save_dir, filename="loss_history.json")
            save_best_test_loss_json(effective_history, self.save_dir, filename="best_test_loss.json")
            if self.save_all_components:
                plot_all_loss_components(
                    effective_history,
                    self.save_dir,
                    filename="所有损失项详细图.png",
                    num_pde_losses=self.num_pde_losses,
                    num_bc_losses=self.num_bc_losses,
                    pde_loss_names=self.pde_loss_names,
                    bc_loss_names=self.bc_loss_names,
                )
        except Exception:
            pass

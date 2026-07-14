from __future__ import annotations

import json

import torch
from torch.utils.data import DataLoader, TensorDataset

from flow_interpolation.training.callbacks import ValidationCallback
from flow_interpolation.training.checkpoints import (
    find_latest_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
)
from flow_interpolation.training.runs import (
    create_workdir,
    load_training_arguments,
    save_training_config,
)
from flow_interpolation.utils.training import EMA, calculate_mfu


def test_create_workdir_never_reuses_a_run_name(tmp_path) -> None:
    first = create_workdir(tmp_path, run_name="experiment")
    second = create_workdir(tmp_path, run_name="experiment")

    assert first.name == "experiment"
    assert second.name == "experiment-01"
    for relative_path in ("artifacts", "checkpoints", "samples/model", "samples/ema", "tensorboard"):
        assert (first / relative_path).is_dir()


def test_config_snapshots_do_not_overwrite(tmp_path) -> None:
    workdir = create_workdir(tmp_path, run_name="config-test")
    initial = save_training_config(workdir, {"seed": 1}, resolved={"device": "cpu"})
    resumed = save_training_config(
        workdir,
        {"seed": 2},
        resolved={"device": "cpu"},
        resumed_from=workdir / "checkpoints/step_000000010.pt",
    )

    assert initial.name == "config.json"
    assert resumed != initial
    assert json.loads(initial.read_text())["arguments"]["seed"] == 1
    assert json.loads(resumed.read_text())["arguments"]["seed"] == 2
    assert load_training_arguments(workdir)["seed"] == 1


def test_full_checkpoint_restores_model_optimizer_ema_step_and_rng(tmp_path) -> None:
    torch.manual_seed(11)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ema = EMA(model, decay=0.9)

    loss = model(torch.randn(4, 3)).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    ema.update()

    expected_model = {name: value.detach().clone() for name, value in model.state_dict().items()}
    expected_ema = [value.detach().clone() for value in ema.shadow_params]
    path = tmp_path / "checkpoints/step_000000012.pt"
    save_training_checkpoint(
        path,
        step=12,
        model=model,
        optimizer=optimizer,
        ema=ema,
    )
    expected_random = torch.rand(4)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for shadow in ema.shadow_params:
            shadow.zero_()
    torch.manual_seed(999)

    result = load_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        ema=ema,
        device="cpu",
    )

    assert result.step == 12
    assert result.full_state
    assert find_latest_checkpoint(tmp_path) == path.resolve()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected_model[name])
    for value, expected in zip(ema.shadow_params, expected_ema, strict=True):
        torch.testing.assert_close(value, expected)
    torch.testing.assert_close(torch.rand(4), expected_random)


def test_legacy_model_checkpoint_is_still_loadable(tmp_path) -> None:
    model = torch.nn.Linear(2, 2)
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    path = tmp_path / "model_step_7.pth"
    torch.save(model.state_dict(), path)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    result = load_training_checkpoint(
        path,
        model=model,
        optimizer=None,
        ema=None,
        device="cpu",
    )

    assert result.step == 7
    assert not result.full_state
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name])


def test_mfu_uses_forward_plus_backward_training_flops() -> None:
    assert calculate_mfu(50e12, batch_time=1.0, peak_flops=100e12) == 50.0


class _TinyFlow(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(2, 2)

    def forward(self, x, conditioning, t):
        del conditioning, t
        return self.projection(x)


def test_validation_restores_training_mode() -> None:
    model = _TinyFlow().train()
    loader = DataLoader(TensorDataset(torch.randn(4, 2)), batch_size=2)

    def criterion(module, data, conditioning):
        return module(data, conditioning, torch.zeros(data.shape[0])).square().mean()

    callback = ValidationCallback(
        model=model,
        criterion=criterion,
        device=torch.device("cpu"),
        num_iterations=1,
        call_every=1,
        amp_dtype=None,
        log_enabled=False,
    )
    callback(1, loader)

    assert model.training

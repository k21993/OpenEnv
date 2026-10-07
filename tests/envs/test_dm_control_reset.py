"""Reset contract tests for the DM Control environment."""

import asyncio

import pytest
from envs.dm_control_env.server.dm_control_environment import DMControlEnvironment


class _FakeDMControlBackend:
    def reset(self) -> object:
        return object()


def _make_environment(monkeypatch: pytest.MonkeyPatch) -> DMControlEnvironment:
    env = DMControlEnvironment()
    env._env = _FakeDMControlBackend()
    monkeypatch.setattr(
        env,
        "_get_observation",
        lambda _time_step, include_pixels=False: object(),
    )
    return env


def test_reset_preserves_requested_episode_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _make_environment(monkeypatch)

    env.reset(episode_id="requested-episode")

    assert env.state.episode_id == "requested-episode"


def test_reset_async_preserves_requested_episode_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _make_environment(monkeypatch)

    asyncio.run(env.reset_async(episode_id="requested-async-episode"))

    assert env.state.episode_id == "requested-async-episode"


def test_reset_preserves_empty_episode_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _make_environment(monkeypatch)

    env.reset(episode_id="")

    assert env.state.episode_id == ""


def test_reset_preserves_existing_positional_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _make_environment(monkeypatch)

    env.reset(
        "cartpole",
        "balance",
        7,
        True,
        episode_id="positional-compatible",
    )

    assert env.state.episode_id == "positional-compatible"
    assert env._include_pixels is True

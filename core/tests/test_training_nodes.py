"""Training as graph nodes: the canvas -> graph mapping, and the runners that drive the service."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from inline_core.config import models_dir
from inline_core.errors import ComponentError
from inline_core.graph.loader_runners import LoraRef
from inline_core.graph.schema import Node, PortKind, port_satisfies
from inline_core.models.training.runner import (
    CAPTION,
    LOAD_DATASET,
    TRAIN_LORA,
    CaptionRunner,
    Dataset,
    LoadDatasetRunner,
    TrainingBridge,
    TrainLoraRunner,
)
from inline_core.runtime.context import CancelToken, ExecutionContext
from inline_core.runtime.progress import CollectingEmitter
from inline_core.studio import moodboard as mb
from inline_core.studio.graph_build import build_workflow_graph
from inline_core.studio.schema import apply_schema


@pytest.fixture
def conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    apply_schema(c)
    c.execute("INSERT INTO project (id, name, created_at, updated_at) VALUES ('p', 'Proj', 0, 0)")
    return c


def _ctx() -> tuple[ExecutionContext, CollectingEmitter]:
    emitter = CollectingEmitter()
    ctx = ExecutionContext(
        run_id="r1", policy=_NullPolicy(), emitter=emitter, cancel=CancelToken()
    )
    return ctx, emitter


class _NullPolicy:
    """The training nodes never place a tensor; the executor only needs the attribute to exist."""


class FakeTraining:
    """Stands in for the service. Its runs advance one poll at a time, like the real one."""

    def __init__(self, *, steps: int = 2, final: str = "done") -> None:
        self.datasets = [{"id": "d1", "name": "Maya"}]
        self.captioned: list[tuple[str, bool, str | None]] = []
        self.started: list[tuple[str, dict[str, Any]]] = []
        self.cancelled: list[str] = []
        self._polls = 0
        self._steps = steps
        self._final = final

    def list_datasets(self) -> list[dict[str, Any]]:
        return self.datasets

    def auto_caption(self, dataset_id: str, overwrite: bool, model: str | None) -> None:
        self.captioned.append((dataset_id, overwrite, model))

    def start(self, dataset_id: str, hyperparams: dict[str, Any]) -> dict[str, Any]:
        self.started.append((dataset_id, hyperparams))
        return {"id": "run-1"}

    def cancel(self, run_id: str) -> None:
        self.cancelled.append(run_id)
        self._final = "interrupted"
        self._polls = self._steps

    def status(self, run_id: str) -> dict[str, Any]:
        self._polls += 1
        if self._polls > self._steps:
            return {
                "status": self._final,
                "outputLoraPath": "loras/maya.safetensors" if self._final == "done" else "",
                "error": "" if self._final == "done" else "stopped",
            }
        return {
            "status": "training",
            "step": self._polls,
            "totalSteps": self._steps,
            "progressFraction": self._polls / self._steps,
            "progressStatus": "training",
        }


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("inline_core.models.training.runner.POLL_SECONDS", 0)


# --- the canvas -> graph mapping -----------------------------------------------------------------


def test_a_second_payload_wire_adds_rather_than_replaces(conn: sqlite3.Connection) -> None:
    """Write .char takes `payload[]`, and only IMAGE_LIST counted as a list: a graph that trained
    an adapter and compiled references wrote whichever was wired last, dropping the training."""
    from inline_core.graph.schema import is_list_kind
    from inline_core.models.character import runner as cr

    descriptors = {d.type: d for d in (cr.WRITE, cr.ATTACH, cr.COMPILE_REFS)}

    def is_list_port(node_type: str, port_id: str) -> bool:
        descriptor = descriptors.get(node_type)
        if descriptor is None:
            return False
        port = next((p for p in descriptor.inputs if p.id == port_id), None)
        return port is not None and is_list_kind(port.kind)

    adapter = mb.add_core_node(conn, "character/adapter", 0, 0)
    refs = mb.add_core_node(conn, "character/references", 0, 0)
    write = mb.add_core_node(conn, "character/write", 0, 0)
    mb.create_connector(conn, adapter["id"], write["id"], "payload", "payloads")
    mb.create_connector(conn, refs["id"], write["id"], "payload", "payloads")

    graph, _target = build_workflow_graph(conn, Path("/tmp"), write["id"], is_list_port)
    node = next(n for n in graph["nodes"] if n["id"] == write["id"])
    wired = [e["from"] for e in node["inputs"]["payloads"]]

    # Set, not list: a payload is filed under its own arch and kind, so unlike an image list the
    # order it arrives in carries no meaning (and both connectors share a created_at here).
    assert set(wired) == {adapter["id"], refs["id"]}, "both payloads reach the file"


def test_a_wired_training_chain_becomes_a_graph(conn: sqlite3.Connection) -> None:
    dataset = mb.add_train_dataset(conn, 0, 0)
    caption = mb.add_caption(conn, 0, 0)
    trainer = mb.add_trainer(conn, 0, 0)
    mb.update_item(conn, dataset["id"], {"data": {"datasetId": "d1"}})
    mb.update_item(conn, caption["id"], {"data": {"overwrite": True}})
    mb.update_item(conn, trainer["id"], {"data": {"hyperparams": {"rank": 32}}})
    mb.create_connector(conn, dataset["id"], caption["id"], "dataset", "dataset")
    mb.create_connector(conn, caption["id"], trainer["id"], "dataset", "dataset")

    graph, target = build_workflow_graph(conn, Path("/tmp"), trainer["id"], lambda _t, _p: False)
    nodes = {n["id"]: n for n in graph["nodes"]}

    assert target == trainer["id"]
    assert nodes[dataset["id"]]["type"] == "train/dataset"
    assert nodes[dataset["id"]]["params"]["dataset_id"] == "d1"
    assert nodes[caption["id"]]["params"]["overwrite"] is True
    assert nodes[trainer["id"]]["params"]["hyperparams"] == {"rank": 32}
    # The wiring is what makes one Run sequential, so the edges have to survive the mapping.
    assert nodes[caption["id"]]["inputs"]["dataset"][0]["from"] == dataset["id"]
    assert nodes[trainer["id"]]["inputs"]["dataset"][0]["from"] == caption["id"]


def test_a_loss_graph_node_is_not_part_of_the_graph(conn: sqlite3.Connection) -> None:
    """It plots a run rather than producing one, so it has no runner and never runs."""
    dataset = mb.add_train_dataset(conn, 0, 0)
    trainer = mb.add_trainer(conn, 0, 0)
    loss = mb.add_loss_graph(conn, 0, 0)
    mb.update_item(conn, dataset["id"], {"data": {"datasetId": "d1"}})
    mb.create_connector(conn, dataset["id"], trainer["id"], "dataset", "dataset")
    mb.create_connector(conn, trainer["id"], loss["id"], "run", "run")

    graph, _ = build_workflow_graph(conn, Path("/tmp"), trainer["id"], lambda _t, _p: False)
    assert loss["id"] not in {n["id"] for n in graph["nodes"]}


# --- the port contract ---------------------------------------------------------------------------


def test_a_dataset_only_satisfies_a_dataset_port() -> None:
    assert port_satisfies(PortKind.DATASET, PortKind.DATASET)
    assert not port_satisfies(PortKind.DATASET, PortKind.IMAGE_LIST)
    assert not port_satisfies(PortKind.IMAGE, PortKind.DATASET)


def test_the_training_nodes_are_hidden_from_the_add_menu() -> None:
    """The canvas adds them through its own Training section; served twice they would show twice."""
    assert LOAD_DATASET.hidden and CAPTION.hidden and TRAIN_LORA.hidden


# --- the runners ---------------------------------------------------------------------------------


def test_load_dataset_names_the_dataset_downstream() -> None:
    bridge = TrainingBridge(FakeTraining())
    ctx, _ = _ctx()
    node = Node(id="n", type="train/dataset", params={"dataset_id": "d1"}, inputs={})
    out = LoadDatasetRunner(bridge).run(node, {}, ctx)
    assert out.outputs["dataset"] == Dataset(id="d1", name="Maya")


def test_load_dataset_fails_before_anything_expensive_runs() -> None:
    """Failing here costs nothing; failing after a caption pass costs the caption pass."""
    bridge = TrainingBridge(FakeTraining())
    ctx, _ = _ctx()
    node = Node(id="n", type="train/dataset", params={"dataset_id": "gone"}, inputs={})
    with pytest.raises(ComponentError, match="no longer exists"):
        LoadDatasetRunner(bridge).run(node, {}, ctx)


def test_caption_passes_the_dataset_through() -> None:
    service = FakeTraining()
    ctx, _ = _ctx()
    node = Node(id="n", type="train/caption", params={"overwrite": True}, inputs={})
    out = CaptionRunner(TrainingBridge(service)).run(
        node, {"dataset": [Dataset(id="d1", name="Maya")]}, ctx
    )
    assert service.captioned == [("d1", True, None)]
    assert out.outputs["dataset"] == Dataset(id="d1", name="Maya")


def test_train_lora_blocks_until_the_run_finishes_and_returns_the_adapter() -> None:
    service = FakeTraining(steps=3)
    ctx, emitter = _ctx()
    node = Node(id="n", type="train/lora", params={"hyperparams": {"rank": 8}}, inputs={})
    out = TrainLoraRunner(TrainingBridge(service)).run(
        node, {"dataset": [Dataset(id="d1", name="Maya")]}, ctx
    )
    assert service.started == [("d1", {"rank": 8})]
    # A LoraRef stack, the shape every `lora` input reads - so Attach Adapter and the generation
    # nodes take the adapter straight off the wire instead of being handed a string they ignore.
    # Absolute, because the run row stores it relative to the models root and a consumer opens the
    # value as given: unresolved it pointed under the server's CWD, where nothing was.
    assert out.outputs["lora"] == (
        LoraRef(file=str(models_dir() / "loras/maya.safetensors"), strength=1.0),
    )
    # The loss curve is wired, not looked up: the run id rides its own port.
    assert out.outputs["metrics"] == "run-1"
    # Steps reach the graph's own stream, which is what makes one Run legible while training.
    assert [e.step for e in emitter.events] == [1, 2, 3]


def test_a_failed_run_fails_the_node_with_the_trainer_s_own_error() -> None:
    service = FakeTraining(steps=1, final="failed")
    ctx, _ = _ctx()
    node = Node(id="n", type="train/lora", params={}, inputs={})
    with pytest.raises(ComponentError, match="stopped"):
        TrainLoraRunner(TrainingBridge(service)).run(
            node, {"dataset": [Dataset(id="d1", name="Maya")]}, ctx
        )


def test_cancelling_the_graph_reaches_the_trainer_once() -> None:
    """Cancel is a SIGTERM the trainer answers by flushing a resumable checkpoint, so the node keeps
    polling afterwards - but must not re-send it every second while that flush happens."""
    service = FakeTraining(steps=5)
    ctx, _ = _ctx()
    ctx.cancel.cancel()
    node = Node(id="n", type="train/lora", params={}, inputs={})
    with pytest.raises(ComponentError):
        TrainLoraRunner(TrainingBridge(service)).run(
            node, {"dataset": [Dataset(id="d1", name="Maya")]}, ctx
        )
    assert service.cancelled == ["run-1"]


def test_train_lora_needs_a_dataset() -> None:
    ctx, _ = _ctx()
    node = Node(id="n", type="train/lora", params={}, inputs={})
    with pytest.raises(ComponentError, match="needs a dataset"):
        TrainLoraRunner(TrainingBridge(FakeTraining())).run(node, {}, ctx)


def test_the_canvas_graph_validates_against_the_registered_descriptors(
    conn: sqlite3.Connection,
) -> None:
    """The wiring is only real if the port ids the canvas emits are the ones the nodes declare. A
    mismatch here is a run that dies at submit, and neither side's unit tests would see it."""
    from inline_core.graph.registry import Registry
    from inline_core.graph.schema import parse_graph
    from inline_core.graph.validate import validate
    from inline_core.models.training.runner import register_training_nodes

    dataset = mb.add_train_dataset(conn, 0, 0)
    caption = mb.add_caption(conn, 0, 0)
    trainer = mb.add_trainer(conn, 0, 0)
    mb.update_item(conn, dataset["id"], {"data": {"datasetId": "d1"}})
    mb.create_connector(conn, dataset["id"], caption["id"], "dataset", "dataset")
    mb.create_connector(conn, caption["id"], trainer["id"], "dataset", "dataset")

    registry = Registry()
    register_training_nodes(registry, FakeTraining())
    graph_dict, target = build_workflow_graph(
        conn, Path("/tmp"), trainer["id"], lambda _t, _p: False
    )
    validate(parse_graph(graph_dict), target, registry)

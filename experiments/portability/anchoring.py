from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from portallib import PortalModel
from safetensors.torch import save_file

from data import SEQUENCE_TASKS, Example
from learner import LossBatch, SequenceLearner, extend_portal, tensor_hash
from repair_alignment import PUBLISHED_TASKS, file_hash, product_loss
from run import Cycle


@dataclass(frozen=True)
class Reference:
    task: str
    boundary: str
    vector: torch.Tensor
    factors: dict

    def tensors(self) -> dict[str, torch.Tensor]:
        tensors = {"vector": self.vector}
        for surface, projections in self.factors.items():
            for (layer, module), pair in projections.items():
                for name, tensor in zip(("A", "B"), pair, strict=True):
                    tensors[f"{surface}.{layer}.{module}.{name}"] = tensor
        return tensors


def anchor_schedule(tasks: list[str], steps: int, seed: int) -> list[dict]:
    result = []
    for stage in range(len(SEQUENCE_TASKS)):
        cycle = Cycle(tasks, seed + stage * 97)
        previous = SEQUENCE_TASKS[:stage]
        for step in range(1, steps + 1):
            result.append(
                {
                    "stage": stage,
                    "step": step,
                    "published": cycle.draw(1)[0],
                    "previous": previous[(step - 1) % len(previous)]
                    if previous
                    else None,
                }
            )
    return result


class FunctionalAnchors:
    def __init__(
        self, source: PortalModel, targets: dict[str, PortalModel], protocol: dict
    ):
        if tuple(source.config.tasks) != PUBLISHED_TASKS + SEQUENCE_TASKS:
            raise ValueError("ANCHOR_SOURCE_TASK_ORDER_MISMATCH")
        if set(targets) != {"qwen4", "mistral7"}:
            raise ValueError("ANCHOR_BOTH_TARGET_SURFACES_REQUIRED")
        self.source = source
        self.protocol = protocol
        self.train_tasks = tuple(protocol["train_tasks"])
        self.holdout_tasks = tuple(protocol["holdout_tasks"])
        self.surfaces = {"qwen8": source}
        self.references: dict[str, Reference] = {}
        self.reference_hashes: dict[str, str] = {}
        self.forward_calls = {"reference": 0, "student": 0}
        for name, target in targets.items():
            if tuple(target.config.tasks) != PUBLISHED_TASKS:
                raise ValueError(f"ANCHOR_TARGET_TASK_ORDER_MISMATCH: {name}")
            if tensor_hash(target.core.state_dict()) != tensor_hash(
                source.core.state_dict()
            ):
                raise ValueError(f"ANCHOR_INITIAL_CORE_MISMATCH: {name}")
            if not torch.equal(
                target.task_latents, source.task_latents[: len(PUBLISHED_TASKS)]
            ):
                raise ValueError(f"ANCHOR_PUBLISHED_VECTOR_MISMATCH: {name}")
            device = next(source.core.parameters()).device
            with torch.random.fork_rng(
                devices=[device.index or 0] if device.type == "cuda" else []
            ):
                target = extend_portal(target, SEQUENCE_TASKS).to(device)
            target.alignment.requires_grad_(False)
            target.task_latents.requires_grad_(False)
            self.surfaces[name] = PortalModel(
                target.config,
                target.task_latents,
                core=source.core,
                alignment=target.alignment,
            )
            self.surfaces[name].task_latents.requires_grad_(False)
        if any(
            parameter.device != next(source.core.parameters()).device
            for model in self.surfaces.values()
            for parameter in model.parameters()
        ):
            raise ValueError("ANCHOR_ALL_SURFACES_MUST_SHARE_SOURCE_DEVICE")
        if any(
            parameter.requires_grad
            for model in self.surfaces.values()
            for parameter in model.alignment.parameters()
        ):
            raise ValueError("ANCHOR_ALIGNMENT_MUST_BE_FROZEN")
        if source.task_latents.requires_grad:
            raise ValueError("ANCHOR_TABLE_MUST_BE_FROZEN")
        if any(
            parameter.dtype != torch.float32
            for model in self.surfaces.values()
            for parameter in model.parameters()
        ):
            raise ValueError("ANCHOR_FLOAT32_PARAMETERS_REQUIRED")
        self.frozen_geometry = self.geometry_hashes()
        for task in PUBLISHED_TASKS:
            self._capture(
                task, source.task_latents[source.config.tasks.index(task)], "initial"
            )

    def geometry_hashes(self) -> dict:
        return {
            "alignments": {
                name: tensor_hash(model.alignment.state_dict())
                for name, model in self.surfaces.items()
            },
            "published_vectors": tensor_hash(
                {"vectors": self.source.task_latents[: len(PUBLISHED_TASKS)]}
            ),
        }

    @torch.no_grad()
    def _capture(self, task: str, vector: torch.Tensor, boundary: str) -> None:
        if task in self.references:
            raise ValueError(f"ANCHOR_REFERENCE_ALREADY_FROZEN: {task}")
        frozen_vector = vector.detach().clone()
        factors = {
            name: {
                key: tuple(value.detach().clone() for value in pair)
                for key, pair in model(frozen_vector).items()
            }
            for name, model in self.surfaces.items()
        }
        reference = Reference(task, boundary, frozen_vector, factors)
        self.forward_calls["reference"] += len(self.surfaces)
        self.references[task] = reference
        self.reference_hashes[task] = tensor_hash(reference.tensors())

    def capture_acquired(self, task: str, vector: torch.Tensor, stage: int) -> None:
        if stage not in (0, 1) or task != SEQUENCE_TASKS[stage]:
            raise ValueError("ANCHOR_ONLY_ACQUIRED_A_OR_B_CAN_BE_ADDED")
        if set(self.references) != set(PUBLISHED_TASKS + SEQUENCE_TASKS[:stage]):
            raise ValueError("ANCHOR_ACQUISITION_BOUNDARY_ORDER_MISMATCH")
        self._capture(task, vector, f"after_{task[-1]}")

    def verify_frozen(self) -> dict:
        if self.geometry_hashes() != self.frozen_geometry:
            raise ValueError("ANCHOR_FROZEN_ALIGNMENT_OR_PUBLISHED_VECTOR_CHANGED")
        for task, reference in self.references.items():
            if any(value.requires_grad for value in reference.tensors().values()):
                raise ValueError(f"ANCHOR_REFERENCE_HAS_GRADIENT: {task}")
            if tensor_hash(reference.tensors()) != self.reference_hashes[task]:
                raise ValueError(f"ANCHOR_REFERENCE_CHANGED: {task}")
        return {
            "geometry": self.frozen_geometry,
            "references": self.reference_hashes.copy(),
        }

    def task_loss(self, task: str) -> tuple[torch.Tensor, dict[str, float]]:
        reference = self.references[task]
        losses = {}
        for name, model in self.surfaces.items():
            losses[name] = product_loss(
                model(reference.vector),
                reference.factors[name],
                self.protocol["normalization_epsilon"],
                model.config.alpha / model.config.rank,
            )
            self.forward_calls["student"] += 1
        return torch.stack(list(losses.values())).mean(), {
            name: float(value.detach()) for name, value in losses.items()
        }

    def loss(
        self, published: str, previous: str | None, stage: int
    ) -> tuple[torch.Tensor, dict]:
        if published not in self.train_tasks:
            raise ValueError(f"ANCHOR_TRAIN_TASK_REQUIRED: {published}")
        if (stage == 0 and previous is not None) or (
            stage > 0 and previous not in SEQUENCE_TASKS[:stage]
        ):
            raise ValueError("ANCHOR_PREVIOUS_ACQUIRED_SKILL_REQUIRED")
        tasks = [published] if previous is None else [published, previous]
        pools, surfaces = [], {}
        for task in tasks:
            value, by_surface = self.task_loss(task)
            pools.append(value)
            surfaces[task] = by_surface
        loss = torch.stack(pools).mean()
        if not torch.isfinite(loss) or float(loss.detach()) < -1e-9:
            raise FloatingPointError("ANCHOR_NONFINITE_OR_NEGATIVE_LOSS")
        return loss, {
            "loss": float(loss.detach()),
            "published": published,
            "previous": previous,
            "surface_task_losses": surfaces,
        }

    @torch.no_grad()
    def diagnostics(self) -> dict:
        self.verify_frozen()
        result = {}
        groups = {
            "train": self.train_tasks,
            "holdout": self.holdout_tasks,
            "acquired": tuple(
                task for task in SEQUENCE_TASKS if task in self.references
            ),
        }
        for split, tasks in groups.items():
            measured = {task: self.task_loss(task)[1] for task in tasks}
            result[split] = {
                "tasks": measured,
                "surface_means": {
                    name: sum(values[name] for values in measured.values())
                    / len(measured)
                    for name in self.surfaces
                }
                if measured
                else {},
            }
        return result

    def save_reference(self, task: str, directory: Path) -> dict:
        self.verify_frozen()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{task}.safetensors"
        if path.exists():
            raise ValueError(f"ANCHOR_REFERENCE_ARTIFACT_ALREADY_EXISTS: {path}")
        reference = self.references[task]
        tensors = {
            name: value.detach().cpu().contiguous()
            for name, value in reference.tensors().items()
        }
        save_file(
            tensors,
            path,
            metadata={
                "kind": "frozen_function_anchor",
                "task": task,
                "boundary": reference.boundary,
            },
        )
        return {
            "task": task,
            "boundary": reference.boundary,
            "path": str(path),
            "file_sha256": file_hash(path),
            "tensor_sha256": self.reference_hashes[task],
            "bytes": path.stat().st_size,
        }


class AnchoredSequenceLearner(SequenceLearner):
    def __init__(
        self,
        base,
        portal: PortalModel,
        tasks: tuple[str, ...],
        stage: int,
        recipe: dict,
        *,
        targets: dict[str, PortalModel],
        protocol: dict,
        anchor_weight: float,
    ):
        if (
            type(anchor_weight) not in (int, float)
            or not math.isfinite(anchor_weight)
            or anchor_weight not in (0, 1)
        ):
            raise ValueError("ANCHOR_ONLY_PREREGISTERED_ZERO_OR_ONE_WEIGHT")
        super().__init__(base, portal, tasks, stage, recipe)
        self.anchors = FunctionalAnchors(portal, targets, protocol)
        self.anchor_weight = float(anchor_weight)
        self.anchor_tasks: dict | None = None
        self.last_anchor: dict | None = None

    def select_anchor(self, row: dict) -> None:
        if row["stage"] != self.stage:
            raise ValueError("ANCHOR_SCHEDULE_STAGE_MISMATCH")
        self.anchor_tasks = row

    def loss(self, rows: list[Example]) -> LossBatch:
        batch = super().loss(rows)
        if any(row.task != self.tasks[self.stage] for row in rows):
            return batch
        if self.anchor_tasks is None:
            raise ValueError("ANCHOR_SCHEDULE_NOT_SELECTED")
        with torch.set_grad_enabled(self.anchor_weight != 0):
            penalty, receipt = self.anchors.loss(
                self.anchor_tasks["published"],
                self.anchor_tasks["previous"],
                self.stage,
            )
        self.last_anchor = {
            **receipt,
            "weight": self.anchor_weight,
            "supervised_current_loss": float(batch.loss.detach()),
        }
        if self.anchor_weight == 0:
            return batch
        return replace(batch, loss=batch.loss + self.anchor_weight * penalty)

    def step(
        self, current: list[Example], reference: list[Example], replay_weight: float
    ) -> dict:
        self.last_anchor = None
        result = super().step(current, reference, replay_weight)
        if self.last_anchor is None:
            raise ValueError("ANCHOR_CURRENT_BATCH_NOT_MEASURED")
        self.anchor_tasks = None
        return {**result, "anchor": self.last_anchor}

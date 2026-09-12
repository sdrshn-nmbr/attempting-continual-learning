import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import torch
from portallib import ChoiceExample, collate_gold_batch
from portallib.evaluation import PortalInjector

from calibration import source_calibration
from engine import transfer
from runtime import (
    CapabilityBlocked,
    InterruptedRun,
    OutputDirectoryLocked,
    RunLog,
    ensure_gpu,
    load_local_base,
    load_port,
    registry_from,
    select_port,
)

logger = logging.getLogger(__name__)


def smoke(config, log):
    registry = registry_from(config)
    row = select_port(registry, config["model_id"], config["revision"])
    portal = load_port(row, config, log)
    log.metrics["measurements"]["artifact_load"] = {
        "status": "passed",
        "projections": len(portal.generate("rte")),
    }
    blocked = [b for b in row["status"]["blockers"] if b["code"] != "gated_base"]
    if blocked:
        raise CapabilityBlocked(json.dumps(blocked))
    device = ensure_gpu(config, log)
    log.check_stop()
    base = load_local_base(
        row, config["model_path"], device, log, gradient_checkpointing=True
    )
    portal.to(device=device, dtype=torch.float32).requires_grad_(False)
    latent = torch.nn.Parameter(portal.task_latents.mean(dim=0).detach().clone())
    optimizer = torch.optim.AdamW([latent], lr=0.002, weight_decay=0)
    example = ChoiceExample(
        "smoke",
        "Copy the input digits unchanged.\nInput: 1 2 3\nOutput:",
        (" 1 2 3", " 3 2 1"),
        0,
    )
    ids, mask, labels = collate_gold_batch(
        base.tokenizer, [example], max_prompt=96, device=device
    )
    log.event("gpu_smoke_started")
    base.model.train()
    before = latent.detach().clone()
    started = time.perf_counter()
    with PortalInjector(base.model, portal.config) as injector:
        with injector.activate(portal(latent)):
            loss = base.model(
                input_ids=ids, attention_mask=mask, labels=labels, use_cache=False
            ).loss
            if not torch.isfinite(loss):
                raise RuntimeError("smoke_nonfinite_loss")
            loss.backward()
        if (
            latent.grad is None
            or not torch.isfinite(latent.grad).all()
            or latent.grad.norm() == 0
        ):
            raise RuntimeError("smoke_missing_nonzero_finite_latent_gradient")
        gradient_norm = float(latent.grad.norm())
        optimizer.step()
    torch.cuda.synchronize()
    delta = float((latent.detach() - before).norm())
    if delta == 0 or any(
        parameter.grad is not None for parameter in base.model.parameters()
    ):
        raise RuntimeError("smoke_weight_update_or_freeze_contract_failed")
    log.metrics["runtime"]["gpu_execution"] = True
    log.metrics["measurements"]["gpu_forward_backward"] = {
        "status": "passed",
        "loss": float(loss.detach()),
        "latent_grad_norm": gradient_norm,
        "latent_update_l2": delta,
        "base_frozen": True,
        "core_and_alignment_frozen": True,
        "nonreentrant_gradient_checkpointing": True,
        "dtype": "base_bfloat16_latent_float32",
        "seconds": time.perf_counter() - started,
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    log.check_stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    try:
        log = RunLog(args.output_dir, config)
    except OutputDirectoryLocked as exc:
        print(
            json.dumps(
                {
                    "event": "output_directory_lock_rejected",
                    "status": "blocked",
                    "code": "output_directory_locked",
                    "output_dir": str(args.output_dir.resolve()),
                    "error": str(exc),
                }
            ),
            file=sys.stderr,
            flush=True,
        )
        return 2
    try:
        log.install_signal_handlers()
        torch.manual_seed(config.get("seed", 17))
        if config["mode"] == "smoke":
            smoke(config, log)
            log.metrics["claim_scope"] = (
                "Measured capability smoke only; no learning or transfer claim."
            )
        elif config["mode"] == "transfer":
            transfer(config, log)
        elif config["mode"] == "source_calibration":
            source_calibration(config, log)
        else:
            raise ValueError(f"unknown_mode: {config['mode']}")
        log.finish("completed")
        return 0
    except InterruptedRun as exc:
        log.finish("interrupted", error=str(exc))
        return 143
    except CapabilityBlocked as exc:
        log.finish(
            "blocked", blockers=[{"code": "capability_blocked", "detail": str(exc)}]
        )
        return 2
    except Exception as exc:
        logger.exception("portal_run_failed")
        log.finish(
            "failed",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        return 1
    finally:
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import json
import logging
from pathlib import Path

import compression_spectrum as spectrum
import rank_control
import representability as geometry
import torch
from peft import LoraConfig

LOG = logging.getLogger("construct-representation")


def complete_span(vectors, width, generator):
    vectors = vectors.double()
    rows, columns = vectors.shape
    if not columns <= width <= rows or not torch.isfinite(vectors).all():
        raise ValueError("CONSTRUCTION_SPAN_DIMENSIONS_OR_NONFINITE")
    completion = torch.randn(
        rows, width - columns, generator=generator, dtype=torch.float64
    )
    basis = torch.linalg.qr(torch.cat((vectors.double(), completion), dim=1)).Q
    residual = torch.linalg.vector_norm(vectors - basis @ (basis.T @ vectors))
    relative = float(residual / torch.linalg.vector_norm(vectors))
    orthogonality = float(
        (basis.T @ basis - torch.eye(width, dtype=torch.float64)).abs().max()
    )
    if not relative < 1e-10 or not orthogonality < 1e-10:
        raise ValueError("CONSTRUCTION_SPAN_CONTROL_FAILED")
    return basis, {
        "rows": rows,
        "source_columns": columns,
        "canonical_width": width,
        "relative_span_error": relative,
        "orthogonality_error": orthogonality,
    }


def errors(source, actual):
    return geometry.aggregate(
        [
            {
                "path": path,
                **spectrum.residual_statistics(
                    geometry.energy(term), geometry.distance(term, actual[path])
                ),
            }
            for path, term in source.items()
        ]
    )


@torch.no_grad()
def construct(model, source, config, seed):
    targets = list(model.config.resolved_targets())
    if set(source) != {path for _, path in targets}:
        raise ValueError("CONSTRUCTION_SOURCE_MATRIX_SET")
    permitted = set(config["fit"]["trainable"]) | {
        name
        for name in model.state_dict()
        if name.startswith(("alignment.input.", "alignment.output."))
    }
    frozen_before = rank_control.tensor_hash(
        {k: v for k, v in model.state_dict().items() if k not in permitted}
    )
    architecture = model.config.to_dict()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    spans = {}
    for kind in ("input", "output"):
        for group, matrix in getattr(model.alignment, kind).items():
            selected = [
                source[path]
                for target, path in targets
                if getattr(target, kind + "_group") == group
            ]
            vectors = torch.cat(
                [term.a.T if kind == "input" else term.b for term in selected], dim=1
            )
            basis, receipt = complete_span(vectors, model.config.d_core, generator)
            matrix.copy_(basis.T if kind == "input" else basis)
            spans[f"{kind}/{group}"] = receipt
    canonical = {
        path: (
            source[path].a.double() @ model.alignment.input[target.input_group].T,
            model.alignment.output[target.output_group].T
            @ source[path].b.double()
            * (source[path].scale / model.config.scaling),
        )
        for target, path in targets
    }
    fit = geometry.fit_shared_heads(model, canonical, config)
    if fit["status"] != "constructed":
        raise ValueError("CONSTRUCTION_SHARED_HEAD_SOLVE_INCONCLUSIVE")
    frozen_after = rank_control.tensor_hash(
        {k: v for k, v in model.state_dict().items() if k not in permitted}
    )
    if frozen_before != frozen_after or architecture != model.config.to_dict():
        raise ValueError("CONSTRUCTION_CHANGED_FROZEN_STATE_OR_ARCHITECTURE")
    generated = geometry.generated_terms(model, config["fit"]["task"])
    error = errors(source, generated)
    if error["maximum_matrix_relative_frobenius_error"] > 1e-8:
        raise ValueError("CONSTRUCTION_FP64_REPRODUCTION_FAILED")
    return {
        "spans": spans,
        "shared_heads": fit,
        "fp64_weight_error": error,
        "frozen_hidden_network_layer_embeddings_and_all_latents": frozen_before,
        "architecture_unchanged": True,
        "updated_parameters": sorted(permitted),
        "optimizer_steps": 0,
        "alignment_qr_decompositions": len(spans),
    }


def run(config, output):
    if config["kind"] != "learned_alignment_constructive_representability":
        raise ValueError("CONSTRUCTION_PROTOCOL_KIND")
    prior = geometry.verify_protocol(
        Path(config["prior_protocol"]["path"]), config["prior_protocol"]["sha256"]
    )
    output.mkdir(parents=True, exist_ok=False)
    original = prior["configuration"]
    protocol = {
        "config": config,
        "sources": [geometry.pin(Path(__file__))],
        "prior_protocol_verified": True,
        "prior_protocol": prior,
        "runtime": geometry.runtime(),
        "outcomes_observed": False,
    }
    geometry.write_json(output / "protocol.json", protocol)
    torch.set_num_threads(original["numerics"]["cpu_threads"])
    model = geometry.load_native(Path(original["native_initial"]["path"]))
    geometry.verify_source_identity(original, model)
    _, source = spectrum.checkpoint_factors(Path(original["rank8_final"]["path"]))
    result = construct(model, source, original, config["seed"])
    result["fp64_checkpoint"] = geometry.save_native(model, output / "native-fp64")
    model = model.float()
    result["fp32_checkpoint"] = geometry.save_native(model, output / "native")
    restored = geometry.load_native(output / "native", torch.float32)
    terms = geometry.generated_terms(restored, original["fit"]["task"])
    result["fp32_reloaded_weight_error"] = errors(source, terms)
    result["injection"] = geometry.serialization_probe(
        restored.config,
        LoraConfig.from_pretrained(original["rank8_final"]["path"]),
        terms,
        output / "adapter",
        original,
    )
    geometry.verify_protocol(
        Path(config["prior_protocol"]["path"]), config["prior_protocol"]["sha256"]
    )
    if protocol["sources"] != [geometry.pin(Path(__file__))]:
        raise ValueError("CONSTRUCTION_SOURCE_CHANGED")
    result.update(
        status="completed",
        protocol_sha256=geometry.file_hash(output / "protocol.json"),
        fixed_native_task=original["fit"]["task"],
        claim_boundary=config["claim_boundary"],
        behavioral_evaluation=False,
    )
    geometry.write_json(output / "result.json", result)
    LOG.info(
        "CONSTRUCTION_COMPLETED fp64_error=%g fp32_error=%g",
        result["fp64_weight_error"]["relative_frobenius_error"],
        result["fp32_reloaded_weight_error"]["relative_frobenius_error"],
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = json.loads(args.config.read_text())
    try:
        run(config, args.output_dir)
    except Exception as error:
        LOG.exception("CONSTRUCTION_FAILED")
        if args.output_dir.is_dir():
            geometry.write_json(
                args.output_dir / "failure.json",
                {"exception": type(error).__name__, "detail": str(error)},
            )
        raise


if __name__ == "__main__":
    main()

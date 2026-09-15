# Attempting continual learning

**Can a model learn new skills, keep the old ones, and carry them into a different model?**

This repository collects two research tasks conducted in September 2026: their code, experiments, results, and final audits. Replay worked well on the tested tasks. Reliable transfer of newly learned skills remained unproven.

[Read the plain-language guide](research/guide.md) · [Browse the research](research/reading-map.md) · [Download the experiment artifacts](https://github.com/sdrshn-nmbr/attempting-continual-learning/releases/tag/research-snapshot-2026-09-11)

**Verified full portfolio archive (September 13 snapshot):** [R2 location and verification manifest](research/r2-portfolio.json) · [restore instructions](research/reproducing.md#full-portfolio-r2-archive). This private archive preserves all 274,879 files from the later `outputs/portfolio` collection, including the September 12 harness and followthrough runs, in 115 independently restorable parts. It is separate from the curated September 11 GitHub release. Agents should consult this reference before recollecting remote runs or assuming missing local checkpoints were lost. The archived bulk data was removed locally on September 15; only small Git-tracked scripts and reports remain. Restore needed artifacts from R2.

## Start with the evidence

| Question | Observed result | Evidence |
| --- | --- | --- |
| Does replay preserve earlier skills? | LoRA with replay: **191/192 and 192/192** after A → B → C; without replay: **80/192 and 89/192**. Two fixed sequences. | [Sequence comparison](site/evidence/replay-sequences.json) |
| Can replay memory be small? | A pool of 32 old examples preserved **238/240** old-task answers on each of two seeds. | [Bounded replay](site/evidence/replay-small.json) |
| Does it work with generated answers? | Old task: **32/32 with replay versus 1/32 without**; new task: 25/32 versus 27/32. One seed. | [Generated-answer control](site/evidence/replay-generated.json) |
| Do new skills move to other models? | **0/24** passing gain comparisons on the confirmation target panel with frozen old alignments. | [Transfer confirmation](site/evidence/transfer.json) |
| Is a compact successful adapter possible? | Static rank-8 compression retained **850/864**, compared with **862/864** for the full source adapter. | [Independent audit](site/evidence/rank.json) |

These are bounded research results. Four-choice scores, generated answers, geometric agreement, and cross-model transfer are different measurements. The [guide](research/guide.md) explains every lane, including negative results, failed teacher gates, the impossible anchoring threshold, and the distinction between this transfer test and published PorTAL.

## Repository map

| Path | Contents |
| --- | --- |
| [`experiments/portability/`](experiments/portability/) | Sequential learning, replay, target transfer, alignment repair, anchoring, rank controls, teacher calibration. |
| [`compute/experiments/plasticity/`](compute/experiments/plasticity/) | Reset controls, CLINC replay, and bounded replay confirmation. |
| [`compute/experiments/onpolicy/`](compute/experiments/onpolicy/) | Teacher qualification, supervised controls, generated-answer replay, external chaining. |
| [`experiments/active_evidence/`](experiments/active_evidence/) | Evidence selection and consolidation. |
| [`compute/experiments/nla/`](compute/experiments/nla/) | Native activation readout and reconstruction diagnostics. |
| [`compute/experiments/continual/`](compute/experiments/continual/) | Earlier continual-learning pilot and evaluation harness. |
| [`compute/experiments/portal/`](compute/experiments/portal/) | Initial PorTAL calibration and transfer controls. |
| [`experiments/catalog/`](experiments/catalog/) | Pretrained model inventory and export checks; pinned upstream submodule. |
| [`experiments/plasticity/`](experiments/plasticity/) | Earlier portfolio plasticity lane, separate from the later replay harness. |
| [`experiments/distillation/`](experiments/distillation/) | Gated distillation implementation; successful distillation was not demonstrated. |
| [`experiments/runtime/`](experiments/runtime/) · [`compute/`](compute/) | Loader guards, task supervision, collection, and historical cluster recipes. |
| [`experiments/review/`](experiments/review/) | Portfolio review contracts and result audits. |
| [`research/`](research/) | Explanations, bibliography, provenance, and artifact manifests. |
| [`site/`](site/) | Complete static interactive explainer with 19 original evidence documents. |

The original layout, experiment implementations, and configurations are preserved so frozen scientific hashes remain meaningful. Three test files now exclude target/controller configurations when enumerating training arms; the initial Git commit retains their original versions. The release also contains historical executed source snapshots; those can differ from the final source tree.

## Explore locally

```sh
git clone --recurse-submodules https://github.com/sdrshn-nmbr/attempting-continual-learning.git
cd attempting-continual-learning
uv run --no-project python tools/verify.py
uv run --no-project python -m http.server 8000 --directory site
```

Open `http://localhost:8000`. The site has no build step or external runtime dependencies.

The verifier checks source bytes, the 19 evidence copies, source-to-evidence provenance, and the headline arithmetic. It does not rerun GPU training. See [reproducing.md](research/reproducing.md) for CPU tests and restoration of the larger artifacts.

## Provenance and scope

Source tasks: `01a0742f-50ff-7560-9855-beb636f36ccf` and `01a078d2-2090-78e3-934d-24cc655c9e05`. This snapshot was assembled on **September 11, 2026** from their shared research workspace. The final recorded execution audit is dated September 9.

[Snapshot manifest](research/snapshot.json) records each included original path, SHA-256 digest, size, and storage location. Source and curated evidence live in Git; detailed results and trained artifacts live in the release. Environment caches, downloaded pretrained model copies, duplicate transport archives, generated test scratch space, raw process logs, private operational snapshots, and unrelated personal history are excluded. No complete task transcript is published.

Upstream code and model provenance are described in [THIRD_PARTY.md](THIRD_PARTY.md). This repository does not assign a new blanket license to the collected work.

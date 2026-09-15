# Reproducing and verifying the work

There are three different checks: verify this archive, rerun CPU mechanics tests, or repeat the GPU experiments. Passing the first two does not independently reproduce the original model results.

## Verify the checkout

From the repository root:

```sh
uv run --no-project --python 3.12 python tools/verify.py
```

This checks the published source and configuration files against `research/snapshot.json`, matches the 19 curated audits to their original paths and hashes, parses their JSON, and recomputes the sequence and rank-control headline totals. GitHub Actions runs the same dependency-free check.

## Restore detailed evidence and checkpoints

The release contains two archives. Both extract directly into this repository, preserving original relative paths. They use hard links for byte-identical files within each archive.

```sh
gh release download research-snapshot-2026-09-11 \
  --repo sdrshn-nmbr/attempting-continual-learning \
  --dir artifacts
(cd artifacts && shasum -a 256 -c SHA256SUMS)
tar -xzf artifacts/evidence.tar.gz
tar -xzf artifacts/checkpoints.tar.gz
uv run --no-project --python 3.12 python tools/verify.py --artifacts
```

`evidence.tar.gz` includes detailed results, predictions, optimizer-update receipts, comparisons, audit records, environment package inventories, and historical executed source snapshots. `checkpoints.tar.gz` includes retained experiment checkpoint files. These archives do not contain full base models or downloaded pretrained catalog weights.

The Git manifest maps every included original file to its storage location. [releases.json](releases.json) records archive hashes, exact sizes, and direct download links. A normal clone is sufficient to read the guide, inspect all experiment code, verify the curated evidence, and use the interactive explainer.

## Full portfolio R2 archive

The private bucket `continual-learning-archives`, prefix `portfolio/2026-09-13/`, holds the verified full local `outputs/portfolio` snapshot, including later harness and followthrough results. [r2-portfolio.json](r2-portfolio.json) is the machine-readable reference: 274,879 regular files, 221.3 GiB of logical file contents, and 162.7 GiB of compressed archives. All 115 archive objects matched their expected sizes and checksums; a downloaded archive passed SHA-256 verification and a selected restored result matched its original bytes. The source inventory was unchanged at completion. The archive does not change the curated release's scientific claims.

The bucket has no public URL. Its S3 endpoint is recorded in the reference. On the owner's Mac, existing R2 credentials are in `~/.config/axport/r2.env`; do not commit or print them. Other machines require authorized R2 credentials with object-read access to this bucket, supplied using the same `R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY` keys in a protected file.

Archives are independently compressed `tar.zst` parts, with paths relative to `outputs/portfolio`. `files.jsonl.zst` maps every original path to its archive. Each part has a SHA-256 digest and verified R2 size/ETag. Downloading a subset may require its containing part, but never requires downloading every part.

Use Python 3.13+, `uv`, and the `zstd` command. First list the exact paths you need:

```sh
uv run --no-project --python 3.13 --with boto3==1.43.93 python tools/restore_portfolio.py \
  --list --prefix followthrough-20260912/runs/
```

Restore a selected prefix into a **new directory**:

```sh
uv run --no-project --python 3.13 --with boto3==1.43.93 python tools/restore_portfolio.py \
  --prefix followthrough-20260912/runs/EXACT-RUN-NAME/ \
  --destination /path/to/new-restored-portfolio
```

The destination receives the original portfolio-relative paths. Pass `--credentials /path/to/r2.env` on another machine. Omitting `--prefix` restores everything and requires enough free disk space for the entire collection plus one temporary compressed part. Prefer selective restoration on a laptop. The tool verifies each downloaded archive's SHA-256 before extracting it and refuses to overwrite an existing destination. For hard-linked members, select a prefix that also includes the referenced target.

The raw archives preserve original symbolic links. The restore helper relocates absolute links pointing inside the original portfolio root so they point inside the new destination; links outside that root are rejected. File contents are unchanged.

The archived bulk collection was deleted locally on September 15, 2026, after re-verifying the R2 manifest and all 115 archive objects and confirming the local inventory still matched. Only 78 Git-tracked scripts and reports (about 10 MB) remain under `outputs/portfolio`; restore any other needed files from R2. The cleanup receipt is in `r2-portfolio.json`. This archive does not prove that the original remote compute volume still exists. New experiment runs after the snapshot need a new archive; they are not automatically uploaded here.

## Run the portability CPU suite

Restore the evidence archive first: some tests inspect real saved receipts and the frozen data generator. These tests use tiny models and local fixtures rather than launching the original GPU jobs.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  -r experiments/portability/requirements.txt \
  pytest==9.1.1 torch==2.14.0
(cd experiments/portability && \
  uv run --no-project --python ../../.venv/bin/python \
  python -m pytest -q tests)
```

At publication, this suite passed 191 checks with one expected skip on macOS using Python 3.12.11 and the recorded package versions. Two old test files now limit their configuration glob to native/LoRA training arms; added target-evaluation configurations no longer enter that enumeration. Experiment implementations, frozen configurations, result files, and scientific thresholds were not changed. The initial Git commit preserves the original tests, and their original hashes remain in `snapshot.json`.

The separate on-policy configuration test received the equivalent schema correction for the chain controller. The three adjusted test files retain their original hashes in the manifest.

Other lanes retain their own tests and requirements. Run each lane from its own directory in a separate environment: common filenames such as `run.py`, `data.py`, and `config.py` refer to different modules across lanes. Do not recursively run all tests in a single pytest process. The later plasticity lane's dataset dependency differs from the portability training dependency; the repository does not force them into one training environment.

[validation.json](validation.json) records 377 passing parent test cases plus 14 passing subtests across ten suites, with 55 skips: 36 Linux-only runtime checks, 18 NLA tokenizer integration checks without their pinned cache, and one CUDA check. No failures remained in the final CPU runs.

The exact two CPU package inventories are in [environments/](environments/). The on-policy suite enforces `torch==2.10.0`, so use its `requirements-test.txt` in a separate environment. For catalog tests, set `PYTHONPATH` to the absolute `experiments/runtime` directory. For plasticity tests on macOS, set `TMPDIR` to a canonical real directory to avoid `/var` versus `/private/var` path differences in adapter receipts. These setup requirements were verified during assembly; the original runtime guards were kept intact.

## Repeat a GPU experiment

Start with the lane's `protocol.json`, configuration, model registry, and saved environment receipts. These experiments used pinned model revisions, fixed data mappings, fixed budgets, and explicit gates. The historical manifests contain the original cluster context and `/mnt/shared/` paths; the repository is not a turnkey configuration for a new cluster.

The operator must provision the appropriate runtime, obtain the recorded model and dataset revisions, place inputs at the declared paths or create a separately tracked new configuration, and preserve the receipt/hashing contract. Commands under `compute/` that submit jobs affect the historical configured cluster when credentials are present. CPU verification and the site server do not submit jobs.

For exact historical execution, use the source snapshots in the evidence archive, together with their receipts, rather than assuming the final source tree was used by every earlier run. The retained local checkpoints do not establish that every original remote checkpoint is available. Files absent from the original local collection cannot be recovered from this repository.

## How to interpret the results

- A loader check establishes that a runtime can load the model, not that it learned a skill.
- A saved-state check establishes persistence and reload behavior, not broad retention.
- Choice scoring and strict generated answers are separate evaluations.
- Low adapter reconstruction error does not establish behavioral transfer.
- Failed or impossible gates remain visible in the evidence. They are explained in the guide rather than rewritten after the fact.
- The GPU outcomes are historical recorded results. Repository assembly did not retrain the models or rerun the original GPU campaign.

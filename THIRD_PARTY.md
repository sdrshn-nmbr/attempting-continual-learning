# Upstream attribution

## PorTAL

`experiments/catalog/upstream` is a Git submodule of [ramp-public/portallib](https://github.com/ramp-public/portallib), pinned to commit `1a7b8c5b0200c301060bfbceca12653f86d2fa31`. Its Apache-2.0 license, citation, and documentation remain in the submodule. Initialize it with `git submodule update --init`.

Scientific requirements record `portallib==0.2.1` and other lane-specific dependencies. The submodule is an attributed upstream reference, not locally authored research code.

## Natural Language Autoencoders

The native activation injection and reconstruction implementation was adapted from [kitft/natural_language_autoencoders](https://github.com/kitft/natural_language_autoencoders) at commit `0577769b55ad4fdd96d159e983361b97fa4e7331`. The original Apache-2.0 license and adaptation notice are preserved in [`compute/experiments/nla/LICENSE.kitft`](compute/experiments/nla/LICENSE.kitft) and [`NOTICE`](compute/experiments/nla/NOTICE).

## Models, datasets, and checkpoints

Model and dataset repositories, revisions, metadata, and download specifications are retained in each lane's configurations and receipts. Their original licenses and access terms apply. Downloaded pretrained catalog weights and full base models are excluded. The release contains experiment-produced checkpoints and supporting metadata, including adapter files derived from the recorded upstream models; it is not a standalone base-model distribution.

The project makes no new license grant for third-party material. No blanket license has been added for the original research code or narrative.

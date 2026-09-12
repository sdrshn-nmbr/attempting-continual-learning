# Research reading map

This is the saved research inventory: 26 families and 76 unique source URLs. It records what the original tasks consulted and what each source can support. It is not a claim that every bibliography was read or every external experiment replicated. Coverage labels and boundaries below come from the original inventory.

The original machine-readable inventory is [here](../site/evidence/research.json). The interactive explainer includes the later source clarifications used at closeout.

## Base research framing

**Coverage:** opened in original corpus.

Agenda and scope; no single executable learning method.

- [Base agenda](https://labs.baseten.co/agenda)
- [Base manifesto](https://labs.baseten.co/manifesto)
- [Base research collection](https://labs.baseten.co/articles)

## Base post-training and factual memory

**Coverage:** articles opened; key sources refreshed.

Separates data breadth, objective and sampling choices; factual reachability evidence does not settle skill learning.

- [SFT study](https://labs.baseten.co/articles/post-training-science-for-supervised-fine-tuning)
- [Dense, on-policy, or both?](https://labs.baseten.co/articles/dense-on-policy-or-both)
- [Continual factual memory](https://arxiv.org/html/2607.11020v2)

## Amortized cache compaction

**Coverage:** opened in original corpus.

Preserves inference context; persistent parameter learning requires a separate consolidation mechanism.

- [STILL](https://arxiv.org/abs/2606.07878)
- [Neural KV compaction](https://labs.baseten.co/articles/towards-infinite-context-windows-neural-kv-cache-compaction)
- [Repeated KV compaction](https://labs.baseten.co/articles/repeated-kv-cache-for-long-running-agents)

## Natural Language Autoencoders

**Coverage:** article, implementation and local contract inspected.

Model/layer-specific activation verbalization and reconstruction; useful diagnostic, not proof of causal repair or portable skills.

- [NLA article](https://transformer-circuits.pub/2026/nla/)
- [NLA implementation](https://github.com/kitft/natural_language_autoencoders)

## Jacobian and representation geometry

**Coverage:** article, repositories and metadata inspected.

Geometry/correspondence diagnostics require behavioral and causal controls.

- [Jacobian Lens](https://github.com/anthropics/jacobian-lens)
- [Global workspace article](https://transformer-circuits.pub/2026/workspace/)
- [Open lens data](https://github.com/eliebak/open-jlens-data)
- [CKA](https://proceedings.mlr.press/v97/kornblith19a.html)

## Released lens checkpoints

**Coverage:** checkpoint metadata inspected.

Assets are specific to the source model; full Inkling, Inkling-Small, Laguna and Qwen checkpoints are not interchangeable.

- [Inkling lens](https://huggingface.co/PrimeIntellect/inkling-jlens)
- [Laguna lens](https://huggingface.co/PrimeIntellect/Laguna-XS.2-jlens)
- [Qwen2.5 lens](https://huggingface.co/anicka/jlens-qwen2.5-7b-instruct)

## Activation interfaces and recovery

**Coverage:** opened or retrieved in original corpus.

Activation access/recovery evidence does not directly establish binding repair or continual cross-model transfer.

- [Universal Activation Interface](https://arxiv.org/abs/2608.09521)
- [J-Access](https://arxiv.org/abs/2608.11408)
- [Trait persistence comparator](https://github.com/manu-scriptum/j-lens-trait-persistence-test-v2)

## Patching and transfer counterevidence

**Coverage:** retrieved in original corpus.

Counterexamples to interpreting isolated patches or representation similarity as sufficient causal evidence.

- [Multiple-mediator patching interactions](https://arxiv.org/abs/2606.27510)
- [Pythia activation-transfer negative result](https://arxiv.org/abs/2606.03280)

## Goodfire mechanism mixtures and temporal features

**Coverage:** papers/repos inspected in original corpus.

Mechanism decomposition and temporal feature analysis; any retained temporal state must be counted as memory.

- [Mixing Mechanisms](https://github.com/yoavgur/mixing-mechs)
- [Temporal Feature Analysis](https://github.com/eslubana/TemporalFeatureAnalysis)

## Goodfire manifold interventions

**Coverage:** articles/repos opened in original corpus.

Activation interventions and subspaces; not the same mathematical object as PorTAL adapter-generation geometry.

- [Manifold Steering](https://www.goodfire.com/research/manifold-steering)
- [Causalab manifold steering implementation](https://github.com/goodfire-ai/causalab/tree/manifold_steering)
- [SAE Manifold](https://github.com/goodfire-ai/sae-manifold)

## Goodfire reward and reasoning diagnostics

**Coverage:** articles/repos inspected; release coverage varied.

Probe rewards and computation-versus-reasoning tests; a reusable official RLFR training/probe release was not established by this audit.

- [Features as Rewards](https://www.goodfire.com/research/rlfr)
- [Reasoning Theater](https://www.goodfire.com/research/reasoning-theater)
- [Forking Fast](https://github.com/ericb-goodfire/forking-fast)

## Goodfire released features and auxiliary methods

**Coverage:** mixed article/checkpoint inspection.

Model-specific SAEs and feature analysis; no established continual native-adapter combination.

- [R1 interpretability](https://github.com/goodfire-ai/r1-interpretability)
- [DeepSeek R1 SAE](https://huggingface.co/Goodfire/DeepSeek-R1-SAE-l37)
- [Stories in Space](https://www.goodfire.com/research/stories-in-space)
- [Covariance Pooling](https://www.goodfire.com/research/covariance-pooling)

## PorTAL

**Coverage:** primary article, pinned implementation and all seven released-port catalog entries inspected.

Published porting freezes shared core and task vectors while fitting target alignment on task calibration data. Continually updating the shared core adds a separate stability requirement.

- [PorTAL article](https://labs.ramp.com/research/portal-portable-task-adaptation/)
- [portallib](https://github.com/ramp-public/portallib)
- [PorTAL task dataset](https://huggingface.co/datasets/RampPublic/portallib-tasks)

## Other Ramp artifacts

**Coverage:** articles opened in original corpus.

KV communication and an interpretability interface; not persistent parameter learning.

- [Latent Briefing](https://labs.ramp.com/research/latent-briefing-kv-cache/)
- [Steer](https://labs.ramp.com/research/how-we-built-steer/)

## LoRA generation and matrix reconstruction

**Coverage:** originally related-work citations; primary material checked during renewed audit.

LoRAGen directly motivates matching complete adaptation matrices to avoid LoRA factor non-uniqueness. Continual product anchoring is our extension.

- [Text-to-LoRA](https://proceedings.mlr.press/v267/charakorn25a.html)
- [LoRAGen paper](https://openreview.net/pdf?id=mrafO7aTYj)
- [LoRAGen official implementation](https://github.com/tsinghua-fib-lab/LoRAGen)

## Profile/context adapter generation

**Coverage:** citation-only in original corpus.

Adjacent generation methods; not verified evidence of indefinite accumulation or backbone-independent skills.

- [SHINE](https://arxiv.org/abs/2602.06358)
- [Profile-to-PEFT](https://arxiv.org/abs/2510.16282)

## Existing adapter transfer

**Coverage:** related-work citations in original corpus.

Transfer via subspace or manifold mappings; LoRA-X's text-to-image evidence must not be silently generalized to this LLM setting.

- [Cross-LoRA](https://arxiv.org/abs/2508.05232)
- [LoRA-X](https://arxiv.org/abs/2501.16559)
- [CAST](https://arxiv.org/abs/2510.17902)

## Cache matching, repair and editing

**Coverage:** paper/code retrieval with access limitations.

Per-context memory preservation uses retained cache/archive state; does not by itself consolidate model competence.

- [Attention Matching](https://arxiv.org/abs/2602.16284)
- [Attention Matching code](https://github.com/adamzweiger/compaction)
- [RepairKV](https://openreview.net/pdf?id=LsrmZrp7tW)
- [E2RAG](https://github.com/tongxuluo/e2rag)

## Cache selection and reversible eviction

**Coverage:** repository/search coverage varied.

Use repository-level attribution for repository-only claims; these mechanisms concern runtime state.

- [EVOKE](https://github.com/Anyesh/EVOKE)
- [Thought-Aware Attention Matching](https://arxiv.org/abs/2608.12331)
- [SparK](https://ojs.aaai.org/index.php/AAAI/article/view/40466)

## Compression limits and knowledge-to-KV

**Coverage:** retrieved; SR-KI search-only in original corpus.

Compression and selection limits are relevant controls, not evidence of weight consolidation.

- [Pitfalls of KV Compression](https://arxiv.org/abs/2510.00231)
- [Operational Proto-Introspection](https://arxiv.org/abs/2607.18553)
- [SR-KI](https://arxiv.org/abs/2511.06446)

## Doc-to-LoRA

**Coverage:** original search result; full method and limitations refreshed.

Amortizes context distillation into adapters. Chunking grows effective rank to r*K; unrelated-query interference is reported. Not a fixed-capacity continual-portability result.

- [Doc-to-LoRA full paper](https://arxiv.org/html/2602.15902)

## Self-editing and on-policy distillation

**Coverage:** papers, recipe and engineering notes inspected.

Useful consolidation ingredients. Teacher advantage must be demonstrated; local hybrids are not exact reproductions of published systems.

- [SEAL](https://arxiv.org/abs/2506.10943)
- [SDFT v2](https://arxiv.org/html/2601.19897v2)
- [Official SDFT recipe](https://tinker-docs.thinkingmachines.ai/cookbook/recipes/sdft/)
- [SDPO](https://arxiv.org/abs/2601.20802)
- [SDPO engineering notes](https://www.trajectory.ai/field-notes/scaling-sdpo)

## Skill learning and experimental interaction

**Coverage:** opened in original corpus.

Skill descriptions, refinement and interaction are adjacent; in-context skill reuse alone does not meet weight-persistence criteria.

- [Skill-SD](https://arxiv.org/abs/2604.10674)
- [SkillMaster](https://arxiv.org/abs/2605.08693)
- [HExA](https://arxiv.org/abs/2606.29315)

## Latent and subgoal distillation

**Coverage:** retrieved in original corpus.

Privileged latent or subgoal supervision; no demonstrated PorTAL consolidation in the inspected sources.

- [Latent On-Policy Self-Distillation](https://arxiv.org/abs/2608.13040)
- [Subgoal Distillation](https://arxiv.org/abs/2405.02749)

## Continual update architectures and plasticity

**Coverage:** Nested Learning/Agent-Dice originally named in supplied feedback; primary identities checked during audit.

Nested optimization and consensus/curvature-based fusion are established directions. The present fixed-objective pair implements neither architecture.

- [Nested Learning](https://arxiv.org/abs/2512.24695)
- [Agent-Dice](https://arxiv.org/abs/2601.03641)
- [Agent-Dice implementation](https://github.com/Wuzheng02/Agent-Dice)
- [Zyphra plasticity study](https://arxiv.org/abs/2606.24752)

## Evaluation assets

**Coverage:** local contracts; ToolAlpaca suggestion-only for the inspected distillation handoff.

Datasets rather than learning methods; asset selection is separate from a demonstrated algorithm.

- [CLINC paper](https://aclanthology.org/D19-1131/)
- [CLINC dataset](https://huggingface.co/datasets/clinc/clinc_oos)
- [ToolAlpaca dataset](https://huggingface.co/datasets/Ahren09/ToolAlpaca)

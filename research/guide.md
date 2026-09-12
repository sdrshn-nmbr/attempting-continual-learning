# What we tried, what happened, and what it means

The question was whether a model can learn something new, retain it after its context is cleared, and keep learning without losing earlier skills. A harder version asks whether the learned skill survives a change of base model.

Picture three little codebooks: A, B, and C. Train the model on A, then B, then C. Test all three after each stage. Keeping earlier examples in the training mix is **replay**. If A disappears while C improves, that is **forgetting**. The main sequence tests scored four possible answers; a separate experiment required the model to generate its answer.

**The strongest observed result was replay preserving earlier skills on bounded tasks. No experiment here demonstrated open-ended continual learning or reliable transfer of newly learned skills to other base models.**

This guide describes the saved experiments from the two source tasks. The evidence files retain their original bytes, including failed gates and limited claims.

## Practicing old lessons really helped.

**Supported, within these tasks.**

After learning A, then B, then C, a single LoRA adapter with replay scored 191/192 and 192/192. Without replay, it scored 80/192 and 89/192. The successful condition kept old examples in the training mix; the test did not provide those examples in the prompt.

The experiment compared four conditions: ordinary LoRA and native PorTAL, each with and without replay. All four acquired the tasks. Only LoRA with replay passed the complete learning-and-retention rule on both fixed sequences.

Each arm received 288 updates. The comparison checked matched data order, token counts, planned replay exposure, active parameter counts, and saved state. Replay/no-replay pairs were identical through task A, before replay could have any effect. Held-out evaluation used four-choice scoring.

Native PorTAL also benefited from replay: 120→183 correct on sequence 303 and 130→157 on sequence 419, out of 192. But it exceeded the allowed forgetting at an earlier or final stage. Two fixed sequences are useful confirmation, not evidence of indefinite learning or broad superiority.

Evidence: [Three-task replay comparison](../site/evidence/replay-sequences.json)

## A small practice set was enough here.

**Supported, within these tasks.**

On an intent-classification stream, keeping just 32 old examples available for replay preserved 238/240 old-task answers on each of two seeds: 99.2%.

First, replay from the growing history raised old-task accuracy from 12.1% to 97.9%, and from 4.6% to 98.3%. Then a fresh matched comparison tested a pool of 32 examples against the full available history.

The small-pool and full-history conditions used the same current examples and replay counts. Full history scored 235/240 and 236/240; the bounded pool scored 238/240 in both. Saved adapters reproduced all 1,280 predictions after fresh loading, as recorded in the task.

The model was Qwen3.5-4B and the stream contained four intent-classification tasks. The memory budget describes stored replay examples, not all model or training memory. The small numerical advantage does not establish that 32 is universally best. These experiments did not establish lifelong learning.

Evidence: [32-example replay comparison](../site/evidence/replay-small.json)

## Replay also helped when the model had to write the answer.

**Supported, one seed.**

After learning a second symbolic operation, the first operation fell to 1/32 without replay. With replay it finished at 32/32, while the second operation scored 25/32 instead of 27/32.

The model learned from known correct answers, rather than a teacher model’s generated advice. This established a working supervised baseline after several teacher approaches failed.

Both runs used 2,560 supervised tokens. During the second task, replay assigned half the teaching tokens to old examples and half to current examples. Both runs matched through the first task. These were generated-answer tests, distinct from the four-choice sequence tests.

There was one seed and a real tradeoff: replay reduced exposure to the new task. Combining the operations remained poor. This was ordinary supervised learning with an oracle supplying labels; it was not a completed SDFT or SDPO experiment.

Evidence: [Replay with generated answers](../site/evidence/replay-generated.json)

## Learning could continue after a model upgrade.

**Narrow positive.**

After moving from Qwen3-4B to Qwen3-8B, training PorTAL’s shared generator and a new task vector reached 93.75% and 92.19% on two synthetic task variants.

PorTAL uses a small description of a task, a shared generator that turns it into adapter weights, and a model-specific alignment. The experiment asked whether this system could learn something additional after the base model changed.

Sampled old-task accuracy fell by 3.1 and 1.0 percentage points. Ordinary LoRA reached 100% on both variants. The small-vector-only condition was not a repeated success; earlier 1.7B vector-only trials stayed at 46.9–52.3% development accuracy.

The shared-generator condition used more trainable parameters and old-task replay. Task identity was supplied and answers were scored from four choices. Moving the newly learned skill onward to the untouched 1.7B alignment failed on both variants. This was feasibility, not a method win.

Evidence: [Learning after a model upgrade](../site/evidence/upgrade.json)

## New skills did not reliably move to other models.

**No confirmed transfer.**

The confirmation panel produced zero passing gain comparisons out of 24 across Qwen3-4B, Mistral-7B, Gemma-3-4B, and Gemma-4-E2B.

The panel covered two trained source conditions and three tasks on each of four targets. An isolated Mistral task-C improvement in the first sequence did not repeat: 40/64 against an initial 35/64 became 23/64 against an initial 30/64.

The evaluation used no target updates and no new target calibration examples. It retained both untouched baselines, both source conditions, every target, and the original thresholds. The separate 1.7B evaluations also failed the full three-task transfer test.

A later literature check corrected the interpretation: released PorTAL normally freezes its generator and task vectors, then refits the target alignment with target-task calibration. Keeping old alignments after changing the generator tests a stronger assumption. These results do not refute the published procedure.

Evidence: [Four-target transfer confirmation](../site/evidence/transfer.json) · [Three-task replay comparison](../site/evidence/replay-sequences.json)

## Making the adapters look right did not make transfer work.

**Geometry improved; behavior unresolved.**

Refitting the target alignment reduced old-adapter reconstruction error by about 87% on Qwen4 and 82% on Mistral7. New-task answers did not show a convincing repair-specific transfer benefit.

The idea was to keep the learned shared generator fixed and repair the model-specific connection to it. Fitting used 10 published old-task adapters; four other old tasks were held out to check whether the repair generalized.

The objective matched the full weight change produced by an adapter, rather than its two factor matrices individually. Different factor pairs can describe the same weight change. Identity and deliberately mismatched task controls were included; all 600 CPU fitting steps and reload checks were recorded.

The subsequent audit regraded 12,096 predictions on 288 fresh inputs per task. Qwen4 changed by 0, +8, +5 correct answers versus unchanged alignment; Mistral changed by +10, −12, −7. Released target adapters provided privileged target information. This hybrid was not a literal PorTAL reproduction, and a failed repair does not prove the learned information vanished.

Evidence: [Alignment repair audit](../site/evidence/repair.json) · [Complete saved research inventory](../site/evidence/research.json)

## Protecting old adapter weights had a tradeoff.

**No overall improvement.**

An added preservation penalty improved task A from 26/32 to 29/32, but reduced B from 31/32 to 28/32 and C from 32/32 to 30/32. The total fell from 89/96 to 87/96.

Instead of repairing alignment afterward, the experiment penalized changes to old generated adapters while the shared generator learned new tasks. Both conditions already used replay; the only comparison was penalty off versus on.

Both arms completed the fixed 288 updates. Old task vectors and model alignments stayed fixed. The audit checked 44 saved reference/checkpoint files and regraded 2,880 predictions. Adapter geometry was better preserved, but the overall score difference’s interval included zero.

Task C had an impossible gain requirement, discussed below. That defect cannot count against the algorithm. Separate feasible retention failures still occurred, and target evaluation was not run. This pair does not test every penalty strength, an interaction with replay, or a new learning architecture.

Evidence: [Preservation penalty audit](../site/evidence/anchor.json)

## A compact successful adapter exists.

**Supported representation result.**

Compressing the successful source adapter to rank 8 preserved 850/864 correct answers (98.38%), compared with 862/864 (99.77%) for the full adapter.

The researchers asked whether PorTAL’s smaller rank alone explains its gap. They took an already successful LoRA adapter and mathematically compressed its weight changes after training. No optimizer steps were used for this control.

The independent audit regraded 3,456 predictions. The full adapter exactly repeated its prior fresh-input result. Compression preserved all initial choices. The final full adapter scored 287/288, 287/288, 288/288 on A/B/C; rank 8 scored 288/288, 280/288, 282/288.

Rank means the number of independent directions allowed in an adapter’s weight change; it is not a count of skills. A compact solution exists on this source model and one mapping. Whether PorTAL’s shared generator can represent it, or training can discover it, remained unanswered. This did not demonstrate free generation or transfer.

Evidence: [Rank-8 compression audit](../site/evidence/rank.json)

## A controller could combine some learned operations.

**Partial success.**

Explicitly chaining the model’s single-operation answers solved 82/128 composed cases (64.06%). Asking for the whole composition in one shot solved 0/128.

Think of a two-step task: transform the input, then transform that result again. The controller made separate calls and passed the model’s own intermediate answer to the next call.

The audit checked the intermediate transitions and ensured hidden correct answers were not substituted. It also compared against the initial model and an oracle last-step control. The final target was at least 90% correct; the chain missed that target.

This establishes useful externally organized use of some learned operations. It does not show that the model learned to compose them internally, and the extra calls are part of the method. The downstream training branch stayed closed.

Evidence: [External chaining audit](../site/evidence/chain.json)

## Picking informative examples did not give a confirmed advantage.

**Confirmation failed.**

The observed advantage over random selection shrank from 5.3 percentage points to 1.9 points on confirmation. The confirmation interval ran from about −5.1 to +8.5 points.

A selector chose queries to eliminate possible rules in a controlled environment. The portfolio compared random and informed evidence selection, including observed-label and inferred-label replay variants.

Budgets and real parameter-update checks were matched. The selector had privileged knowledge of the possible rule family, making it a strong test of whether better evidence could help this learner.

The next-skill learning requirement also failed. This is no confirmed active-acquisition benefit for the tested recipe, not proof that choosing examples intelligently never helps.

Evidence: [Choosing informative examples](../site/evidence/active.json)

## Resetting parts of the learner hurt retention.

**Negative for this reset recipe.**

New intent tasks were acquired at roughly 98–100%, but the initial pilot’s final old-task accuracy was 29.2% with continuous training, 19.6% with optimizer resets, and 0% with selective parameter resets.

Plasticity means remaining able to learn. The proposal was that resetting stale training state or selected parameters could restore that ability.

The four-method pilot completed 512 updates. But the model was already acquiring the new tasks, so the experiment did not demonstrate a sustained loss of learning ability that a reset repaired.

Later replay studies exposed a more useful issue: without replay, many old examples were classified as the newest labels. Practice corrected that behavior. This does not uniquely identify whether old information was erased or simply became hard to use.

Evidence: [Initial reset experiment](../site/evidence/plasticity.json) · [32-example replay comparison](../site/evidence/replay-small.json)

## The proposed teachers were not ready to teach.

**Prerequisite failed.**

No SDFT-versus-SDPO training comparison was completed. The teacher first had to produce correct answers under the exact teaching format, and the attempted teachers failed that check.

SDFT uses a demonstration-informed teacher to guide the student’s own attempts. SDPO uses feedback or privileged information to improve that guidance. Both can update weights; choosing either still requires a competent teacher for the actual task.

Early composition teachers scored 0/48, or 1/48 with altered instructions. A later 9B teacher still scored 0/48 compositions; an answer-conditioned 4B teacher scored 25/80 even with the correct answer supplied. The final learned Qwen3-8B adapter was excellent at choosing among answers, but strict generated validation responses scored only 1/32, 5/32, and 2/32 on A/B/C.

Stopping mattered: the last teacher often wrote correct first digits and continued. A post-hoc check found 26/32, 20/32, and 32/32 correct three-digit prefixes, but this did not satisfy or replace the original full-response rule. No distillation followed. This distinguishes answer knowledge, response formatting, and a usable teaching signal.

Evidence: [Early teacher composition audit](../site/evidence/teacher-early.json) · [Learned Qwen3-8B teacher qualification](../site/evidence/teacher.json) · [Teacher answer and stopping audit](../site/evidence/teacher-ending.json)

## Readable internal descriptions were not a qualified repair tool.

**Diagnostic not qualified.**

The NLA branch did not produce evidence of reliable diagnosis or repair of continual-learning failures. There were no learning updates in its qualification runs.

An NLA translates an internal activation into text and reconstructs the activation from that text. Topic recognition alone is too weak: identifying “keys and colors” is different from knowing who has which key after an exchange.

One task report described an early broad-topic readout of 8/8 and only 1/4 reliably distinguished changed-fact pairs. The saved stricter native-fidelity diagnostic failed its control gates, including scorable reconstruction checks. These are different checks, not contradictory learning scores.

Model and layer compatibility, shuffled and empty controls, and causal intervention tests all matter. The branch did not establish that fluent explanations locate erased knowledge, that arbitrary activations can be read by prompting Claude, or that an interpreter transfers to a new model.

Evidence: [Native activation readout diagnostic](../site/evidence/nla.json)

## Six released ports were evaluated. Inkling remained unmeasured.

**Baseline tools verified.**

Qwen3 1.7B/4B/8B, Mistral7B, Gemma3 4B, and Gemma4 E2B completed the common catalog evaluation. These used released adapters and zero learning updates.

Each smaller model used the same 128 examples across BoolQ, HellaSwag, WinoGrande, and CommonsenseQA, with native adapter, export, and reload checks. Mistral had the largest observed change: 72.7% to 80.5%.

None established a confirmed advantage after accounting for the six comparisons. The seventh planned port, Inkling, downloaded and passed content verification but encountered host-memory problems during loading.

The first Inkling load was killed by the container’s host-memory limit; its replacement lacked the local cache. A later guarded attempt crossed the 2 TiB stop threshold before scoring and was terminated. That is an infrastructure result, not zero model accuracy or evidence that Inkling cannot learn. Qwen loader parity and process-cleanup controls were separately verified.

Evidence: [Six-model catalog audit](../site/evidence/catalog.json) · [Guarded Inkling attempt verification](../site/evidence/inkling.json)

## Reading the gates correctly

A gate is a rule declared before judging a result. Passing a loader or checkpoint check means the machinery worked; it does not mean the model learned. A good training score does not establish retention, transfer, or generated-answer competence.

One anchoring task started at 19/32 but required an improvement of 50 percentage points. Even a perfect final score could improve by only 40.625 points. That gate was impossible, so its failure cannot count against the algorithm. The stored audit keeps the original false flag and records the defect; other feasible retention failures remained.

The target-transfer tests kept old model alignments frozen after changing the shared generator. Published PorTAL normally refits a target alignment using calibration data. The local test therefore examined a stronger assumption, rather than reproducing the published target-adaptation procedure.

The next useful research question was whether the shared generator could represent a known successful compressed adapter. That experiment was proposed but had not been executed at closeout.

## The tangible products

- Experiment implementations, frozen configurations, data fixtures, and tests in `compute/` and `experiments/`.
- Nineteen directly readable evidence documents in `site/evidence/`.
- The full research-family inventory and source links in [reading-map.md](reading-map.md).
- Structured run results, prediction traces, audits, executed source snapshots, and trained checkpoints in the [snapshot release](https://github.com/sdrshn-nmbr/attempting-continual-learning/releases/tag/research-snapshot-2026-09-11).
- A dependency-free interactive explainer in `site/`.

These products are a research archive. The recorded GPU runs were not repeated when this repository was assembled.

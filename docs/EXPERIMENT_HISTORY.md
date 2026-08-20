# Experiment history

The release came from the `script/train-minimax-music-rvq-encoder` SimpleTuner branch. Commit identifiers below refer to that repository.

## Replanner

| Commit | Change | Result |
|---|---|---|
| `903de9c3b` | Initial flow-matching latent replanner with MERT, CLAP, and RVQ streams | Established the model and evaluation path. |
| `44a94166c` | MERT span masking and learned nulls | Reduced dependence on exact MERT frames. |
| `d7d3a98a3` | Identity examples | Added a high-correlation structural task. |
| `d532c029d` | Restoration task | Introduced damaged-input to clean-target training. |
| `87c89b893` | DDPM objective arm | Worked, but underperformed flow and bridge objectives. |
| `2a95ac9ba` | Source-derived MERT/CLAP, degraded latent stream, bridge objective | Made restoration inputs internally consistent. |
| `f7205c67a` | Explicit task conditioning | Separated transfer, identity, restore, and null behavior. |
| `6bded9f22` | Task-aware bridge endpoints | Removed endpoint inconsistencies between tasks. |
| `16a2c2ff5` | In-context editing | Added the complete degraded DAV sequence as reference tokens. |
| `a3c15b0df` | Correct per-frame stream placement | Applied streams to the target segment before context concatenation. |
| `40fffa4bc` | Residual-weighted loss and residual-cosine evaluation | Made correction quality visible independently of passthrough. |
| `495b577db` | Bridge context reference plus SR3 stream | Produced the selected hybrid architecture. |

## Refiner ablation

All arms used the same restoration task and holdout protocol.

| Arm | Holdout result |
|---|---|
| No degraded latent stream | weak; failed to beat passthrough |
| DDPM | residual cosine about 0.36 |
| Flow matching plus degraded stream | residual cosine about 0.64 |
| Bridge transport | residual cosine 0.7015 |
| Bridge plus context plus degraded stream, step 1,000 | residual cosine 0.7195; diagonal cosine 0.9479 |
| Same hybrid, step 2,000 | residual cosine 0.7105; diagonal cosine 0.9405 |

Checkpoint 1,000 was selected before the later decline.

## Preference branch

The preference branch did not produce the release model.

| Commit | Change | Verdict |
|---|---|---|
| `7dfed3f30` | On-policy flow-DPO rejects | Unbounded reject-side incentive damaged the sampler while base losses remained normal. |
| `b9ced1e81` | Positive margin cap, lower beta and weight | Prevented runaway but still degraded sampler quality. |
| `27f2acc44` | Source latents as static rejects | Anti-passthrough experiment; not part of v0.10. |

The acceptance rule is sampler evaluation, not training loss. Preference objectives that raise error on samples from the model's own trajectory can damage flow sampling without changing the ordinary task losses.

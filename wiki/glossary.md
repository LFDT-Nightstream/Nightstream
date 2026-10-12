# Glossary

| Term | Meaning | Current owner |
|---|---|---|
| CCS | Matrices and a polynomial defining the computation relation | `neo-ccs` |
| CE | A commitment with public values and claimed matrix evaluations | `neo-ccs`, `nightstream/src/folding/claims.rs` |
| PiCCS | Reduce fresh and running claims to a common evaluation point | `nightstream/src/folding/pi_ccs.rs` |
| PiRLC | Combine 17 claims with strong-set challenges | `nightstream/src/folding/pi_rlc.rs` |
| PiDEC | Decompose the result into 16 low-norm claims | `nightstream/src/folding/pi_dec.rs` |
| NIFS | The composed non-interactive C/R/D fold | `nightstream/src/folding/compose.rs` |
| F′ | Application execution plus recursive fold verification and state binding | `formal/nightstream-fprime/NightstreamFPrime/Lifecycle/Stage1` |
| Strong set | Degree-54 ring vectors with coefficients in `{−2,…,2}` | `PiRlcSampler` specification and `neo-reductions/src/common/pi_rlc_sampler.rs` |
| Wide reduction | Four field draws interpreted in base p and reduced modulo `5^54` | The single production PiRLC sampler |
| Ajtai commitment | Low-norm, ring-linear commitment under the fixed key | `neo-ajtai` |
| Phi81 | `X^54 + X^27 + 1`, defining the commitment ring | `neo-math` |
| Pad / Eval_K | Evaluation of the padded witness | Running opening check |
| Matrix / Eval_A | Evaluation of a matrix applied to the witness | Separate check for each of 14 slots |
| Terminal verification | Check the fresh relation and running openings against the expected final state | `nightstream/src/lifecycle/verify.rs` |

The [protocol pages](protocol/index.md) explain the paper construction. The
[security model](../formal/nightstream-fprime/SECURITY_MODEL.md) states
what the implementation proofs establish.

# Glossary

| Term | Meaning | Current owner |
|---|---|---|
| CCS | Matrices and a polynomial defining the computation relation | `neo-ccs` |
| CE | A commitment with public values and claimed matrix evaluations | `neo-ccs`, `nightstream/src/folding/claims.rs` |
| PiCCS | Reduce fresh and running claims to a common evaluation point | `nightstream/src/folding/pi_ccs.rs` |
| PiRLC | Combine the sources of a fold. A light fold has 2 sources, with the weights ρ and 1. A refresh fold has 17 sources, with strong-set challenges | `nightstream/src/folding/pi_rlc.rs` |
| PiDEC | In a refresh fold, decompose the accumulator into 16 low-norm claims | `nightstream/src/folding/pi_dec.rs` |
| NIFS | The light fold and the refresh fold | `nightstream/src/folding/option_b.rs` |
| F′ | Application execution plus recursive fold verification and state binding, in a light and a refresh step circuit | `formal/nightstream-fprime/NightstreamFPrime/Lifecycle/OptionB` |
| Strong set | Degree-54 ring vectors with coefficients in `{−2,…,2}` | `PiRlcSampler` specification and `neo-reductions/src/common/pi_rlc_sampler.rs` |
| Wide reduction | Four field draws interpreted in base p and reduced modulo `5^54` | The single production PiRLC sampler |
| Ajtai commitment | Low-norm, ring-linear commitment under the fixed key | `neo-ajtai` |
| Phi81 | `X^54 + X^27 + 1`, defining the commitment ring | `neo-math` |
| Pad / Eval_K | Evaluation of the padded witness | The slot check of `OptionB::verify`, by PiDEC recomposition |
| Matrix / Eval_A | Evaluation of a matrix applied to the witness, in four families | The slot check of `OptionB::verify`, by PiDEC recomposition |
| Terminal verification | Check the fresh instance and the claims of both slots against the expected final state | `nightstream/src/lifecycle/option_b/mod.rs` |

[SPEC.md](../SPEC.md) describes the protocol that Nightstream runs. The
[security model](../formal/nightstream-fprime/SECURITY_MODEL.md) states
what the implementation proofs establish.

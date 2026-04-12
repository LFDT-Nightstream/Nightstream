# TODO

## Neo-Fold-Next Breakdown

| Area | Paper-backed rule | Why it matters | Next action |
|---|---|---|---|
| `shard` ownership | `neo-fold-next` should have one obvious shard prover owner that scripts the flow without absorbing all protocol internals. | Auditors need one place to trace shard proving end to end. | Refactor `neo-fold-next` around a small `shard::prover` plus owner modules below it. |
| `main_lane` | SuperNeo `Pi_CCS -> Pi_RLC -> Pi_DEC` remains the core fold owner for CE-family claims at one shared point. | This is the paper-core folding path; it should stay explicit and not get hidden behind generic wrappers. | Define the minimal `main_lane` API and proof artifact boundary. |
| `twist_shout` owner | Twist/Shout should start as an extension owner module, not as a promised fold lane. | The papers justify projection from virtual claims to committed evaluation obligations, but not arbitrary merging into the main lane. | Create a `twist_shout` owner that outputs explicit obligation families. |
| Fold admissibility | Fold together only claims of the same relation at the same evaluation point. | "Project everything into the main lane" is not paper-backed unless there is an explicit reduction proving it. | Partition obligations by relation family and evaluation point before folding. |
| Twist/Shout integration | Use Twist/Shout first as projection protocols, then fold only whatever becomes an admissible CE family. | This is the most straightforward sound path from the papers. | Specify which outputs become main-lane CE claims, separate fold families, or final obligations. |
| SumCheck security accounting | SumCheck soundness is probabilistic and must be aggregated globally across all SumCheck-family uses; do not treat thousands of SumChecks as isolated. SuperNeo already uses union-bound style accumulation for bad events and composition-level negligible loss. | Security accounting is wrong if each local SumCheck error is mentally ignored after the call returns. | Define one protocol-level `eps_sumcheck_total` for `neo-fold-next`, or a tighter grouped theorem if available. |
| Lean validation | The right formal target is a bridge layer between SuperNeo and Twist/Shout, not a literal transcription of implementation stages. | We need theorem-backed architecture, not just code structure. | Add a new Lean bridge package proving projection, family partition, fold admissibility, and shard composition. |
| `build_superneo_ring_forms` API footprint | The helper is acceptable as a deterministic algebra routine, but it should not stay a wide root-level compatibility surface forever. | It is not a transcript/soundness risk, but it is extra `neo-ccs` API surface and a large materialized form object. | Keep it for now, then remove the root re-export once downstream imports are localized and add a direct equivalence test against `y_ring = Z · M_j^T · χ(r)` before any deeper refactor. |

## Security Note

- SuperNeo explicitly treats SumCheck soundness as probabilistic and combines bad-event probabilities by union bound.
- For `neo-fold-next`, that means all SumCheck-family calls across `main_lane`, `twist_shout`, and any opening-reduction path must roll into one protocol-level accounting story.
- There is no runtime "reject if error > threshold" mechanism; the protocol must be parameterized so the aggregated bound is negligible before we call it secure.

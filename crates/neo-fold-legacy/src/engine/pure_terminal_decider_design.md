# Pure-accumulator terminal decider — design note

**Status:** design note, not implementation. Produced under the council's
explicit instruction: *"if the 'in-circuit check of the running
accumulator' requires essentially implementing a CCS/Spartan verifier
inside the relation, stop and produce a design note instead of building
a fake check. No digest-only authority."*

## Goal

Build a terminal-decider R1CS synthesis that emits **strictly fewer**
rows than [`synthesize_last_step_terminal_r1cs`](decider.rs), by not
emitting the last encoded-F' step's full F' shell. The minimal honest
relation would emit only:

- The terminal NIFS.V fold (already in `emit_terminal_fold`).
- The terminal-latest link (binding the fresh CCS instance's public
  input to the public image's `x_out`).
- The public-image pins.
- **A real in-circuit check that the final running accumulator's
  witness `Z` satisfies the F' CCS structure under the canonical
  preprocessing.**

The last bullet is the load-bearing piece.

## Why this is hard right now

The chain's inductive soundness threads through each `enc(F'_i)`: that
instance's witness embeds the in-circuit NIFS.V trace of the (i−1)th
fold. If the LAST encoded F' instance is proved to be a valid CCS
instance, the entire chain is transitively bound. That proof of
validity is exactly what the terminal SNARK (Spartan over the encoded F'
relation) is supposed to provide — it's the inductive base case.

`synthesize_last_step_terminal_r1cs` discharges that base case by
*re-emitting the last F' step's R1CS shell* in the decider relation. A
downstream Spartan over THAT relation then sumcheck-verifies the
shell. This is sound but not minimal: the relation grows by one F'
shell at the last step.

A pure-accumulator-only relation would skip emitting the F' shell.
But then "the running's witness Z satisfies the F' CCS structure" is
no longer a tautology the relation enforces — *something else inside
the relation has to assert it*. The candidates are:

| Option | What it does | Verdict |
|---|---|---|
| Pin only the running's commitment + claim digest | Cryptographically gates that the prover *claims* the right commitment, but does not gate that there exists a valid Z behind it. | **Digest-only authority** — forbidden by the council. |
| Embed a sumcheck (Π_CCS-style) verifier in the relation, taking a prover-supplied transcript that proves "Z exists with `log.commit(Z) == claim.c` and Z satisfies F' CCS" | This is exactly what Spartan-in-circuit does. Sound; verifies the accumulator without re-emitting the F' shell. | **Requires the Spartan-in-circuit primitive.** Not yet integrated in this crate; `decider::prove`/`verify` are still `Unsupported`. |
| Use a special commitment scheme with cheap openings | Doesn't apply: SuperNeo uses Ajtai, whose opening is a sumcheck-style proof. | Not available without re-engineering the commitment scheme. |

The middle option *is* what HyperNova §6.3 Construction 2 calls for at
the terminal decider — Spartan compresses the final accumulator. Until
the Spartan-in-circuit verifier primitive lands in this crate, there
is no honest pure-accumulator terminal-decider implementation: any
in-circuit "check" we could write today would either re-emit the F'
shell (which is what `synthesize_last_step_terminal_r1cs` already
does) or fall into digest-only authority.

## Why the existing `emit_terminal_fold` is not enough on its own

`emit_terminal_fold` runs the terminal NIFS.V (`Π_CCS → Π_RLC → Π_DEC`)
in-circuit. Π_CCS already contains a sumcheck-based verifier that
proves the **input** CE claims fold correctly into the **output** CE
claim. That gives "given valid R_{N−1} and valid fresh_N, R_N is a
valid CE claim." It does **not** give "R_{N−1} is itself a valid CE
claim" — that has to come from somewhere else.

In `synthesize_last_step_terminal_r1cs`, the F' shell at step N gives
it: the shell verifies in-circuit that step N's fresh CCS instance is
valid, and inside that instance's witness is the in-circuit NIFS.V
trace of step N−1's fold, transitively grounding the chain.

In a pure-accumulator decider, there is no F' shell. The grounding has
to come from a direct in-circuit verification of the final running
accumulator's validity. That's the Spartan-in-circuit primitive.

## Recommendation

**Defer the pure-accumulator terminal decider** until the Spartan
integration (the council's "Step 3" — `decider::prove`/`verify`) lands.
The natural order is:

1. **First** land Spartan terminal compression on top of
   `synthesize_last_step_terminal_r1cs`. This gives the production a
   real verifier-side SNARK without history replay, even though the
   relation still emits one F' shell.
2. **Then** build the pure-accumulator decider as a smaller relation
   that uses the Spartan-in-circuit primitive (newly available from
   step 1) to verify the running accumulator without re-emitting the
   F' shell.

Doing the pure-accumulator decider *before* Spartan would require
either inventing the Spartan-in-circuit primitive in isolation (large,
risky, and orphaned without a SNARK to consume it) or emitting a
"check" that violates the council's no-digest-only-authority rule.

## What this slice *does not* change

- `synthesize_last_step_terminal_r1cs` stays as the only terminal
  synthesis function and continues to be steady-state O(1).
- `decider::prove` / `decider::verify` remain `Unsupported`.
- `synthesize_pure_terminal_accumulator_r1cs` is not added.
- No tampered/red-team test surface is added for a non-existent
  pure-terminal relation.

## What re-opens this design

The two unblockers for the pure-accumulator decider:

1. A Spartan-in-circuit sumcheck verifier primitive available as a
   building block in this crate (presumably arrives with the Spartan
   terminal compression integration).
2. A second IVC frontend or app shape whose terminal cost makes the
   "skip the last F' shell" win meaningful in practice. With only the
   Fibonacci F' shape today, the F' shell's row count is the same
   per-step constant the chain already pays; trimming one F' shell
   from the terminal is a constant-factor improvement on a relation
   that is already steady-state O(1).

When either of those lands, revisit this note and the Step 2/3
ordering.

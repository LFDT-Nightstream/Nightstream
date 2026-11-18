# Light compression of the WASM sizes

## Status

Accepted by the owner on 2026-11-16 (Version B):

- The WASM sizes have big light steps, of 664 to 800 rows, and no proof of
  them reaches a refresh fold.
- The final proof of these sizes has two claims: slot 0 and the fresh
  instance. The Akita key holds the light relation only.
- The generic circuits, such as the Poseidon2 hash chain, keep the refresh
  fold and the three claims.
- Each memory circuit whose proofs have at most 65 steps, `N · S_max ≤ 65`,
  gets the two-claim proof, not only the WASM sizes.
- A second memory fingerprint, for longer proofs, is a separate change.
- An empty refresh step is necessary only for proofs of more than about
  45,000 instructions.

## Problem

The refresh step circuit runs the same WASM program as the light step
circuit, and it also checks the refresh fold. With three claims, both
relations must fit the Akita key of `2^19` blocks
([`akita-compression.md`](./akita-compression.md)). The refresh fold check
alone takes most of that key, so the WASM step is capped at a small number
of instructions, and folding costs more for each instruction. With big
steps, the memory gate already ends each proof of a size before the refresh
fold of step 66 ([`wasm-fixed-sizes.md`](./wasm-fixed-sizes.md), item 6).
So the refresh relation is in the key only for proofs that do not exist.

## SuperNeo

SuperNeo v1.2 fixes the light and the refresh fold. It fixes no step size
and no schedule of refresh folds. The bounded-depth schedule refreshes a
slot at the cap (`Spec.Folding.Schedule.productionLightFoldCap`), and the
compression of the final proof is a project design
([`akita-compression.md`](./akita-compression.md)).

## Decision

[SPEC.md](../SPEC.md) Algorithms 9, 10 and 11 are the rule.

1. **The mode.** A final proof has the light mode when each proof of its
   circuit has at most `cap + 1 = 65` steps. Then the first step is the base
   step and each later step is a light fold into slot 0 below the cap, so
   slot 1 stays the default slot at depth 0 and the light circuit runs last
   (`Export.BoundedDepth.RunCompleteness.light_run_complete`). Rust selects
   the light mode for a memory circuit whose geometry has `N · S_max ≤ 65`,
   because the memory verifier accepts at most `S_max` segments of `N`
   steps (SPEC.md Algorithm 8). The WASM sizes have this property
   (`Spec.Nebula.WasmSize.steps_le_light`). Each other circuit keeps both
   slots.
2. **The verifier takes the mode from its circuit.** A proof cannot select
   it. The verifier rejects a proof that carries the slots of the other
   mode, and in the light mode a proof with `pc ≠ 1`. It builds slot 1 as
   the default slot at depth 0 from its own relation and never reads it from
   the proof (`Export.BoundedDepth.FinalVerifier.ModeChecks`).
3. **Two claims.** The argument has slot 0 and the fresh instance of the
   light circuit, so the Akita key holds the light relation alone. Slot 0
   stays in the argument at depth 0, because the layout needs a claim with
   a table (`Spec.Compression.Rust.layoutNew_light`).
4. **One proof form for each mode.** A light final proof has the tag
   `NS-OPTIONB-FIN-L` and carries slot 0 alone. One verifier sees one mode,
   so the transcript needs no mode word: the sponge calls fix the run among
   the runs of one statement shape
   (`Export.BoundedDepth.CompressionSponge.calls_injective`).
5. **The prover does not refuse a refresh fold.** No proof of a light
   circuit that the memory verifier accepts reaches one. `compress` refuses
   a Stage 1 proof whose slot 1 is not the default slot at depth 0 or whose
   last circuit is the refresh circuit, so a proof that made a refresh fold
   gives an error before the commitment.
6. **The refresh premises stay.** The generic circuits keep the refresh
   fold, so `[HN-A1]` for the refresh fold and its knowledge error stay in
   the [security model](../formal/nightstream-fprime/SECURITY_MODEL.md).

### Consequences

- The light final verifier has the bound of the final verifier with both
  slots (`Export.BoundedDepth.FinalVerifier.soundness`,
  `Export.BoundedDepth.FinalVerifier.Mode`).
- A WASM proof has at most 30 steps. A longer proof needs more segments,
  and the first change is a second memory fingerprint. At more than 65
  steps a proof reaches a refresh fold, and the light mode does not apply.

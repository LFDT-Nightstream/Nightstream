# neo-fold-clean

A paper-faithful, audit-first re-implementation of the SuperNeo IVC integrator.

## What this crate is

The integrator on top of the protocol-primitive crates (`neo-reductions`,
`neo-ccs`, `neo-ajtai`, `neo-transcript`, `neo-math`, `neo-params`, `spartan2`).
It owns:

- The three reductions Π_CCS / Π_RLC / Π_DEC, in paper order, as a thin facade
  over the engine in `neo-reductions`.
- Hypernova Construction 2's augmented function F' and the IVC step.
- The Spartan terminal-compression contract.
- One generic `direct_ccs` frontend.

The protocol math itself lives in the sibling crates and is *not* reimplemented
here.

## What this crate is not

- Not a new fold engine. The optimized engine in `neo-reductions` is reused
  unchanged; the paper-exact engine likewise stays where it is.
- Not a frontend playground. There is one frontend (direct CCS). VM frontends
  are out of scope until this crate is the canonical integrator.
- Not a perf surface. Diagnostics, traces, and shape probes are kept out of
  the protocol path.

## Design rules (non-negotiable)

1. **Paper names.** Code identifiers track the paper symbols defined in
   [`paper/mod.rs`](src/paper/mod.rs). When in doubt, the glossary wins.
2. **Step-down.** Every public function reads top-to-bottom as a sequence of
   named operations. Each step decomposes the same way one level down. No
   spaghetti control flow in protocol paths.
3. **Poseidon2 only** in protocol-binding paths. No mixed hash families.
4. **Digests are compression, never authority.** Every carried digest is
   re-derived by the verifier from authoritative inputs.
5. **Files ≤ 1500 lines.** If a file grows past that, the design is wrong, not
   the line count.
6. **Type the gaps.** A protocol step that isn't proof-complete must be
   represented in the type system, not as a runtime string error.

## Open Questions

Unresolved protocol questions live in [`open-questions/`](open-questions/).

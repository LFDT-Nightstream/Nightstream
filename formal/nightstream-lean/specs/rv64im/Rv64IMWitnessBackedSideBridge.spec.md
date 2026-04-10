# Rv64IMWitnessBackedSideBridge

## Purpose

- **What it is**: The end-state theorem-facing contract for the RV64IM Nightstream
  side bridge after the current hybrid native seam is replaced by a
  witness-backed relation and then compiled by a succinct backend.
- **What it is not**: It is not the accepted-artifact replay layer, the current
  native side-terminal witness artifact, or a permission to weaken the bridge
  theorem to digest consistency.
- **Protocol role**: It fixes the exact public/private split and the exact
  theorem that any later witness-backed bridge relation, recursive wrapper, or
  succinct outer proof must preserve.

## Normative Basis

This owner is constrained by the following local references:

- `./docs/superneo-paper/01_1_Introduction.md`
  - standard compilers apply **after** the folding scheme relation is fixed
  - efficient proof compression with Spartan is a later compiler layer, not the
    definition of the folding relation itself
- `./docs/superneo-paper/06_6_Strong_and_weak_interactive_reductions.md`
  - strong/weak composition requires a stable shared projection `phi`
- `./docs/superneo-paper/07_7_Neo_s_folding_scheme_for_CCS.md`
  - the paper-owned relation is the CE/CCS relation plus
    `Π_CCS -> Π_RLC -> Π_DEC`, not a verifier for a variable-length proof blob
- `./crates/neo-fold-next/specs/riscv-kernel.md`
  - the current concrete RV64IM theorem-facing bridge obligations
- `./crates/neo-fold-next/specs/riscv-recursive-proof.md`
  - the end-state recursive/exported proof architecture

This document is normative only where its requirements are implied by, or are a
conservative specialization of, those references plus the repo-owned RV64IM
bridge theorem.

## Design Goal

The exported RV64IM Nightstream verifier boundary shall eventually validate a
succinct proof of a **fixed witness-backed relation**.

That relation shall own exactly the same semantic bridge theorem currently
checked by the native RV64IM side-claim, side-opening, opening-artifact, and
side-terminal seams, but it shall own those checks as **private witness
constraints**, not as a public verifier replay over shipped native witness
artifacts.

The final public verifier shall not depend on:

- the native side-terminal witness artifact,
- raw side-claim packaged witnesses,
- raw side-opening packaged witnesses,
- raw kernel opening witnesses,
- or any bridge-local digest bundle exported only because the current hybrid
  path happens to carry it.

## Theorem Target

The end-state witness-backed side bridge owns one exact theorem:

> the carried RV64IM side bridge is valid if and only if there exists a private
> witness that realizes the same accepted-opening identity, row-local bridge
> binding, stage linkage, opening provenance, and handoff equality checks that
> the current native RV64IM bridge theorem requires.

In particular, any conforming witness-backed or succinct bridge must preserve:

- the same exact accepted opening identity reused across row projection,
  opening provenance, and bridge binding,
- the same canonical authenticated row-to-chunk routing induced by the carried
  fold schedule,
- the same exact `RootEncode(z_j)` image and row-local acceptance binding for
  each theorem-bearing selected row,
- the same exact prepared-step / exported-row equality checks currently required
  by the bridge theorem,
- the same exact Ajtai opening / commitment consistency checks,
- and the same exact handoff equality to the main residual proof boundary.

It may replace native replay with a private-witness relation or a succinct
compiler, but it may not weaken the theorem to digest consistency or to
"self-consistent carried summaries".

## Public Instance

The public instance of the witness-backed side bridge shall contain only the
minimal authoritative bindings required by the theorem and by external verifier
composition.

At minimum, the public instance shall bind:

- the canonical public proof statement or its canonical digest,
- the canonical side-bundle binding required to identify the accepted side
  surfaces consumed by the bridge theorem,
- the canonical opening-artifact binding required by the accepted opening
  theorem boundary,
- the canonical bridge handoff bindings required to connect the side bridge to
  the main residual proof,
- and the canonical verifier-context / root-parameter binding required by the
  published Nightstream verifier statement.

No field may remain public solely because the current hybrid seam exports it.

No bridge-local digest bundle is theorem-facing by default.

## Private Witness

The private witness shall contain the concrete theorem-bearing objects needed to
realize the bridge theorem.

This includes the objects currently realized through native replay such as:

- the side-claim witness objects,
- the side-opening witness objects,
- the opening-artifact witness/provenance objects,
- the selected-row authenticated payloads and row-local execution/opening
  objects,
- the prepared-step linkage objects,
- and any lower-level packages needed to prove the owned theorem above from the
  public instance.

If a value is purely derived from other witness or public data, it should be
recomputed inside the relation rather than carried as an independent authority.

## Stable Projection Requirement

The RV64IM witness-backed side bridge shall expose one stable projection
`phi_side` for the purposes of strong/weak composition in the sense of
SuperNeo's interactive-reduction framework.

`phi_side` shall project only the authoritative commitment / public-binding data
that identifies the carried bridge instance. It shall not project convenience
digests that can be recomputed from those authoritative values.

Any later reduction or compiler layer that composes above this bridge must bind
the same `phi_side`.

## Compiler Requirements

Any later recursive wrapper or succinct backend compiler over this bridge must
satisfy all of the following:

- it compiles a **fixed relation** with a stable public/private interface,
  rather than directly verifying a variable-length proof object;
- it preserves the exact theorem target above;
- it does not expose the private witness as theorem-facing output;
- it does not introduce additional theorem-facing digest inputs merely for
  convenience;
- and it keeps all protocol-binding transcript inputs canonical and derived from
  authoritative relation data.

If the concrete backend cannot support an unbounded number of bridge components
with one fixed shape, then a conforming implementation must either:

- use a fixed maximum shape with canonical padding and selectors, or
- introduce a recursive wrapper above per-shape compiled relations.

Direct verification of an arbitrary-length side proof by a fixed compiled
circuit is not a required property of this owner.

## Negative Rules

The following are forbidden at this theorem boundary:

- treating digests as authority rather than as compression of authoritative
  objects,
- carrying duplicate authorities for the same fact across the public boundary,
- mixing attacker-chosen helper digests into Fiat-Shamir transcript inputs
  without first recomputing or binding them canonically,
- exposing native witness artifacts as part of the final theorem-facing
  verifier API,
- and using accepted-artifact replay or other prover-side oracle objects as the
  canonical public verifier input.

## Dependency and Consumer Map

- **Depends on**:
  - `./crates/neo-fold-next/specs/riscv-kernel.md`
  - `./crates/neo-fold-next/specs/riscv-recursive-proof.md`
  - `./formal/nightstream-lean/specs/rv64im/Rv64IMBridgeBinding.spec.md`
  - `./formal/nightstream-lean/specs/rv64im/Rv64IMAcceptedArtifactKernelDesignBridgeClosure.spec.md`
  - `./formal/nightstream-lean/specs/rv64im/Rv64IMAcceptedArtifactRootExecutionSemanticsClosure.spec.md`
- **Consumed by**:
  - future Lean interfaces/implementations for the RV64IM witness-backed side
    bridge,
  - RV64IM proof-complete audit closure,
  - and the concrete recursive/backend instantiation docs and code.

## Out of Scope

- fixing the exact succinct backend choice,
- fixing concrete padding bounds or recursion arity,
- re-specifying the inner RV64IM execution semantics,
- or redefining the accepted-artifact oracle/audit path as the public verifier
  boundary.

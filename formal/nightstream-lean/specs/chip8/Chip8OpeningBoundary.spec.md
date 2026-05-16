# Chip8OpeningBoundary Spec

## Purpose

- **What it is**: The theorem-facing opening-manifest contract for the final
  CHIP-8 kernel.
- **Key property**: `kernelOpeningBoundary_conforms`: every kernel-owned opening
  claim references exactly one commitment fixed in `root0`, uses the exact
  commitment-local polynomial registry, appears in the correct kernel/root
  ownership bucket, forbids one global heterogeneous fold carrier on the simple
  boundary, and requires the simple boundary to export no family-local
  fold-bucket carrier at all.
- **Protocol role**: This is the boundary that prevents the kernel from
  overclaiming direct opening access to undeclared commitments or from
  conflating kernel-owned openings with later root-prover openings.

This module classifies opening claims and manifests only. It does **not** own
the checked Shout/Twist proofs themselves, the continuity reduction proof, or
semantic extraction from those claims.

Scope warning:

- this owner is complete only for the `simple` boundary with
  `RootOpeningManifest = ∅`;
- any later combined kernel-plus-root proof with root-owned opening claims must
  introduce its own explicit root opening schema rather than inferring one from
  this owner.

## Target Formulas

### Kernel commitment surface

The final kernel fixes the following opening commitments before any
stage-specific challenge is sampled:

- `C_lane`
- `C_fetch_ra`
- `C_decode_ra`
- `C_alu_ra`
- `C_eq4_ra`
- `C_decode_handoff`
- `C_reg`
- `C_ram`
- `C_rom_table`
- `C_decode_table`
- `C_alu_table`
- `C_eq4_table`

These are the only commitments that may appear in the kernel-owned opening
manifest.

The root prover later creates disjoint commitments, including the per-step
Ajtai commitments inside `PreparedStep`.

### Opening claim shape

Define the theorem-facing opening claim:

$$
\mathrm{OpeningClaim}
:=
(
\mathrm{source},
\mathrm{commitmentId},
\mathrm{point},
\mathrm{polynomialIds},
\mathrm{claimedValues},
\mathrm{digest}
).
$$

The kernel-owned commitment identifier is drawn from:

$$
\mathrm{KernelCommitmentId}
:=
\mathrm{Lane}
\mid
\mathrm{FetchRa}
\mid
\mathrm{DecodeRa}
\mid
\mathrm{AluRa}
\mid
\mathrm{Eq4Ra}
\mid
\mathrm{DecodeHandoff}
\mid
\mathrm{RegTwist}
\mid
\mathrm{RamTwist}
\mid
\mathrm{RomTable}
\mid
\mathrm{DecodeTable}
\mid
\mathrm{AluTable}
\mid
\mathrm{Eq4Table}.
$$

The boundary-wide commitment identifier is then:

$$
\mathrm{CommitmentId}
:=
\mathrm{Kernel}(\mathrm{KernelCommitmentId})
\mid
\mathrm{RootProver}(\_).
$$

### Manifest ownership buckets

The opening boundary is split into two disjoint ownership buckets:

- `KernelOpeningManifest`
- `RootOpeningManifest`

The kernel manifest may reference only the commitments fixed in `root0`.
The root manifest may reference only commitments created after bridge
extraction.

On the simple-kernel boundary, root-side binding is not modeled as a non-empty
root opening manifest. It is carried instead by the exact prepared-step export
and the explicit row-local bridge-binding leaves. Any later combined
kernel-plus-root proof must introduce its own explicit root opening schema
rather than inferring one from this simple boundary.

On that simple boundary, any `OpeningClaim` with `source = root` or
`commitmentId = RootProver(_)` is ill-formed. Any kernel-owned direct opening
claim on that boundary must therefore use `commitmentId = Kernel(cid)` for some
`cid : KernelCommitmentId`.

### Grouping rule

There are two distinct grouping notions:

$$
(\mathrm{commitmentId}, \mathrm{point})
$$

for one family-local direct opening surface, and

$$
(\mathrm{source}, \mathrm{domain}, \mathrm{point})
$$

for one later claim-space reduction bucket, with member claims ordered
canonically by their manifest ordinals inside that bucket.

This boundary owns the first notion. It does not identify witness-space fold
lanes and it does not collapse heterogeneous commitment families into one fold
carrier. On the `simple` boundary, a single global heterogeneous fold carrier
is not merely unidentified; it is non-conforming and must be absent.

If two admissible direct opening claims share the same
`(commitmentId, point)`, they remain distinct only through their exact
`polynomialIds`. This boundary does not require same-surface coalescing, but it
does require deterministic tie-breaking by `polynomialIds` and it forbids two
distinct claims with the same `(commitmentId, point, polynomialIds)`.

### Kernel manifest shape

In this `simple` kernel boundary, the kernel-owned manifest contains exactly the
following direct openings:

- `C_lane @ r_lookup` for the Stage-1 lane columns
  `{PC, KK, NNN_ADDR, NNN_WORD, REG_X, REG_Y, LOOKUP_OUTPUT,
    WritesLookupToX, WritesMemToX, PreservesX, WritesNnnToI,
    IsJump, IsBranch, IsMemOp, X_IDX, Y_IDX, BURST_LAST}`
- `C_fetch_ra @ (r_fetch_addr, r_lookup)`
- `C_decode_ra @ (r_decode_addr, r_lookup)`
- `C_alu_ra @ (r_alu_addr, r_lookup)`
- `C_eq4_ra @ (r_eq4_addr, r_lookup)`
- `C_decode_handoff @ r_lookup` with exact polynomial subset
  `{handoff_uses_y, handoff_reads_ram, handoff_writes_ram}`
- `C_rom_table @ r_fetch_addr`
- `C_decode_table @ r_decode_addr` with exact polynomial subset `0..21`
- `C_alu_table @ r_add8lo_addr`
- `C_eq4_table @ r_eq4_addr`
- `C_lane @ r_twist_cycle` for the Stage-2 lane columns
  `{REG_X, REG_Y, REG_X_NEXT, I_REG, I_NEXT, MEM_VALUE,
    WritesLookupToX, WritesMemToX, PreservesX, WritesNnnToI,
    IsMemOp, X_IDX, Y_IDX, RAM_ADDR}`
- `C_decode_handoff @ r_twist_cycle` with exact polynomial subset
  `{handoff_uses_y, handoff_reads_ram, handoff_writes_ram}`
- `C_reg @ (r_addr_reg, r_twist_cycle)` with exact polynomial subset
  `{RegInc, RegRaX, RegRaY, RegRaI, RegWa}`
- `C_ram @ (r_addr_ram, r_twist_cycle)` with exact polynomial subset
  `{RamInc, RamRa, RamWa}`
- `C_lane @ r_shift` for `{PC, PC_NEXT, X_IDX, IsMemOp, BURST_LAST}`
- `C_lane @ j0_bits` for `{IsMemOp, X_IDX}`
- `C_lane @ j_last_bits` for `{IsMemOp, BURST_LAST}`
- `C_lane @ j_bits` for the exact 23 committed non-fixed lane coordinates of
  every exported semantic row, in canonical `C_lane` registry order:
  `{PC, PC_NEXT, REG_X, REG_Y, REG_X_NEXT, I_REG, I_NEXT, KK, NNN_ADDR,
    NNN_WORD, MEM_VALUE, LOOKUP_OUTPUT, WritesLookupToX, WritesMemToX,
    PreservesX, WritesNnnToI, IsJump, IsBranch, IsMemOp, X_IDX, Y_IDX,
    BURST_LAST, RAM_ADDR}`

No other direct kernel opening claims are admissible in this `simple` kernel
boundary.

These are the committed `C_decode_handoff` columns, not the decode-table output
columns. Stage 1 separately proves
`handoff_uses_y = uses_y_dec`,
`handoff_reads_ram = reads_ram_dec`, and
`handoff_writes_ram = writes_ram_dec`.

So for one semantic prefix of length `N`, the simple kernel manifest contains
exactly `N + 17` direct kernel opening claims: 17 fixed non-row-binding claims
plus one `C_lane @ j_bits` row-binding claim for each exported row.

The local canonical order of those `N + 17` claims is fixed here, not only by
cross-reference:

1. `C_lane @ r_lookup`
2. `C_fetch_ra @ (r_fetch_addr, r_lookup)`
3. `C_decode_ra @ (r_decode_addr, r_lookup)`
4. `C_alu_ra @ (r_alu_addr, r_lookup)`
5. `C_eq4_ra @ (r_eq4_addr, r_lookup)`
6. `C_decode_handoff @ r_lookup`
7. `C_rom_table @ r_fetch_addr`
8. `C_decode_table @ r_decode_addr`
9. `C_alu_table @ r_add8lo_addr`
10. `C_eq4_table @ r_eq4_addr`
11. `C_lane @ r_twist_cycle`
12. `C_decode_handoff @ r_twist_cycle`
13. `C_reg @ (r_addr_reg, r_twist_cycle)`
14. `C_ram @ (r_addr_ram, r_twist_cycle)`
15. `C_lane @ r_shift`
16. `C_lane @ j0_bits`
17. `C_lane @ j_last_bits`
18. then the row-binding claims `C_lane @ j_bits` in strictly increasing
    `row_index`

This local order is the manifest-ordinal order used by digest identity and by
the deterministic claim ordering inside later claim-space reduction buckets.

The `j0_bits` opening is intentionally only the Stage-3 burst-start boundary.
It does not carry `PC(0)`. On the simple boundary, the first-row `pc` is owned
instead by the kernel input contract plus the authenticated first semantic row:
`SimpleKernelChunkInput` fixes `InitialStateMatches(init, first.pre)`, and the
exact trace closure then combines that with the Stage-3 start-boundary rule and
the authenticated row binding.

Here "row-membership proof" means exactly that accepted `C_lane @ j_bits`
row-binding opening together with its exact lower-layer opening witness and
refinement path. There is no separate extra PCS object beyond that authenticated
opening chain.

### Exact exclusions

The opening boundary must also state the exact exclusions:

- `LaneShiftProof` is not an `OpeningClaim`
- virtual `RegVal` and `RamVal` objects are not `OpeningClaim`s
- `KernelOpeningManifest` may not reference any root-prover commitment
- `RootOpeningManifest` may not reference any `C_*` commitment fixed in `root0`
- on the simple-kernel boundary, `RootOpeningManifest = ∅`
- claim-space reduction summaries are not opening claims
- fold carriers are not opening claims
- row-projection summaries are not opening claims
- bridge-binding summaries are not opening claims
- later kernel stages do not reuse Stage-1 openings; each stage opens its own
  direct claims independently
- no split or partial decode-table opening claims at `r_decode_addr` are
  admissible in this `simple` boundary

### Boundary object classification

This opening-boundary owner classifies only direct opening claims and manifest
ownership. For audit clarity:

- soundness-carrying opening path objects:
  - `OpeningClaim`
  - `AcceptedDirectOpening`
  - the lower-layer exact opening witness and refinement path imported by later
    owners
- protocol-binding boundary objects:
  - `KernelOpeningManifest`
  - `RootOpeningManifest`
  - canonical manifest ordering / grouping
- mandatory provenance but not opening claims:
  - row-projection witnesses
  - bridge-binding witnesses
- optional implementation-side carriers / summaries, outside the direct opening
  boundary:
  - claim-space reduction summaries
  - `JointOpeningUnifiedClaimReduction`
  - family-local fold-bucket carriers

On the `simple` boundary, the theorem-facing negative rule is:

- one global heterogeneous fold carrier must be absent;
- the simple boundary exports no family-local fold-bucket carrier at all;
- any later owner that re-enables a family-local fold-bucket carrier must model
  it explicitly outside this simple-boundary theorem surface.

### Future family-local fold-bucket hook

If a later owner chooses to extend the theorem surface with one family-local
fold carrier, this boundary exposes only the minimal positive admissibility
hook:

$$
\mathrm{FamilyLocalFoldBucketConforms}(pts, carrierCommitmentId, carrierPoint, claims)
$$

meaning:

- the carrier summarizes a non-empty claim list
- every summarized direct claim is kernel-owned
- every summarized direct claim uses the same `commitmentId`
- every summarized direct claim uses the same evaluation point value
- every summarized direct claim is already admissible under the kernel opening
  boundary
- the summarized claim list itself has canonical ordering

This is intentionally weaker than a CE / CCS fold theorem. It proves only that
one future bucket stays inside one homogeneous manifest-local opening family at
one common point value. It does not prove that the bucket is itself a new
opening claim or a proved folding lane, and it is not part of the current
`simple` boundary.

In particular, this boundary does **not** by itself fix the stronger
homogeneity discriminants required by the main kernel prose for a real
family-local witness-space fold lane, including:

- commitment setup / committer surface
- encoded witness width
- fold-shape convention
- CE / CCS structure identifier
- commitment-map identifier
- point-shape / evaluation convention
- witness-layout identifier

A later owner must model those additional discriminants explicitly before any
family-local carrier may be treated as a genuine fold authorization.

### Future combined root-side schema

This `simple` boundary intentionally leaves `RootOpeningManifest = ∅`. If a
later combined kernel-plus-root proof becomes part of the theorem surface, that
owner must add an explicit root-side schema that includes:

- the exact root commitment inventory
- the exact root opening-manifest entry kinds
- canonical root-manifest ordering rules
- the exact root-side refinement / provenance path tying exported
  `PreparedStep` artifacts to any root-owned opening claims

This simple boundary does not infer that larger root-side schema on behalf of a
future owner.

This document is therefore complete only for the `simple` boundary with
`RootOpeningManifest = ∅`.

### Canonical manifest ordering

For the `simple` kernel boundary, canonical manifest order is exactly the local
stage order already enumerated above:

1. the 17 fixed non-row-binding claims in that exact order;
2. then the `C_lane @ j_bits` row-binding claims in strictly increasing
   `row_index`.

This local order is the canonical manifest order used for manifest ordinals,
direct-claim digests, and deterministic claim ordering inside later
claim-space reduction buckets.

The generic tuple sort
`(commitmentIdOrder, pointArity, pointCoordinates, polynomialIds)` is not the
simple-boundary kernel-manifest order. If a later owner introduces a non-empty
root manifest or another non-simple manifest surface, that later owner may
define and own a generic sort for that distinct manifest.

The simple boundary still enforces:

- strictly increasing `polynomialIds` in the local registry order
- point coordinates ordered exactly by the commitment's domain convention
- no duplicate `(commitmentId, point, polynomialIds)` entries
- same-surface ties, when present, ordered deterministically by
  `polynomialIds`

### Kernel opening boundary

Define:

$$
\mathrm{KernelOpeningBoundary}(kernelManifest, rootManifest)
$$

to mean:

- every kernel claim references a kernel commitment fixed in `root0`
- every root claim references only root-prover commitments
- the root manifest is empty on the simple-kernel boundary
- the kernel manifest satisfies the exact kernel-shape constraints above
- the kernel manifest uses the exact simple-boundary canonical order above
- any root-manifest ordering rule is vacuous on this boundary because
  `RootOpeningManifest = ∅`

The main theorem target is:

$$
\mathrm{KernelOpeningBoundary}(kernelManifest, rootManifest)
\Longrightarrow
\text{no orphan or mis-owned opening claim exists}.
$$

## Paper Anchors

- **Source**: `./crates/neo-fold-prototype/specs/chip8-kernel.md`
- Anchors:
  - commitment bundle
  - two commitment layers
  - opening boundary
  - canonical manifest ordering

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Chip8/OpeningBoundary.lean` | Kernel/root opening-manifest ownership and conformance theorems |
| `Nightstream/Chip8/OpeningBoundaryInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Commitments | `KernelCommitmentId` | def | Definitional | Enumerates the kernel-owned opening commitment classes fixed in `root0` |
| Commitments | `CommitmentId` | def | Definitional | Enumerates every opening commitment class relevant to the kernel/root boundary |
| Claims | `OpeningClaim` | def | Definitional | Theorem-facing direct opening claim object |
| Manifests | `KernelOpeningManifest` | def | Definitional | Kernel-owned direct opening manifest |
| Manifests | `RootOpeningManifest` | def | Definitional | Root-prover direct opening manifest |
| Boundary | `KernelManifestShape` | def | Definitional | Exact required kernel-owned claim families and points |
| Boundary | `SimpleKernelManifestOrder` | def | Definitional | Exact stage-local canonical order for the simple kernel manifest |
| Boundary | `CanonicalManifestOrder` | def | Definitional | Generic canonical ordering rule reserved for non-simple or later-owned manifests |
| Boundary | `KernelOpeningBoundary` | def | Definitional | Complete kernel/root ownership and conformance predicate |
| Boundary | `SimpleBoundaryGlobalFoldPlanAbsent` | def | Definitional | The simple boundary exports no single global fold plan |
| Boundary | `FamilyLocalFoldBucketConforms` | def | Definitional | Future extension hook: any later family-local bucket carrier must stay inside one kernel commitment family at one common point value |
| Boundary | `LaneShiftSourceOpeningAppearsInManifest` | def | Definitional | Names the required `C_lane @ r_shift` direct opening inside the kernel manifest |
| Theorem | `kernelOpeningBoundary_conforms` | theorem | Theorem-Target | A conforming manifest contains only legal, correctly owned opening claims |
| Theorem | `laneShift_not_openingClaim` | theorem | Theorem-Target | `LaneShiftProof` is not part of either opening manifest |
| Theorem | `laneShiftSourceOpeningAppears_of_kernelManifestShape` | theorem | Theorem-Target | A conforming kernel manifest explicitly contains the Stage-3 `C_lane @ r_shift` source opening |
| Theorem | `kernel_root_commitments_disjoint` | theorem | Theorem-Target | Kernel and root opening commitments remain disjoint |

## Proof Obligations

- This module must model the final kernel/root manifest split explicitly.
- The theorem surface must enumerate the exact kernel opening commitments.
- Canonical manifest ordering and grouping must remain theorem-facing, not only
  implementation prose.
- The simple boundary must state the absence of one global fold plan
  theorem-facing, not only in the main kernel prose.
- The simple boundary must also state the absence of any family-local
  fold-bucket carrier in its accepted theorem surface.
- If a later owner models one optional family-local fold carrier, this module
  must expose only the manifest-local admissibility conditions it can actually
  own.
- `LaneShiftProof` must remain outside the direct opening-manifest type.
- The Stage-3 `C_lane @ r_shift` source opening must still appear explicitly in
  the kernel manifest even though `LaneShiftProof` itself is not an
  `OpeningClaim`.

## Assumption Ledger

- This module does not re-prove PCS batching or final opening verification.
- This module does not prove Shout or Twist checked reductions.
- This module does not prove semantic extraction from opening claims.

## Dependency and Consumer Map

- **Upstream dependencies**:
  - final kernel commitment surface from `chip8-kernel.md`
- **Downstream consumers**:
  - `Nightstream/Chip8/EvidenceCoverage.lean`
  - later Rust-refinement theorems for `KernelOpeningManifest`

## Implementation Plan

1. Define the exact opening claim and manifest objects.
2. Define the kernel/root ownership split and manifest shape.
3. Define canonical ordering and grouping.
4. Prove manifest conformance and exact exclusions.

## Quality Expectations

- Keep this module about direct opening ownership only.
- Specialize to the final kernel commitment inventory.
- Do not smuggle checked Shout/Twist relations into the direct opening manifest.

## Acceptance Criteria

1. `lake build Nightstream.Chip8.OpeningBoundary` succeeds.
2. The theorem surface matches the final kernel/root opening-manifest split.
3. Kernel and root commitment layers remain formally disjoint.
4. No `sorry`.

## Out of Scope

- checked Shout/Twist proof objects
- Stage-3 continuity reduction proofs
- semantic extraction
- final PCS opening verification

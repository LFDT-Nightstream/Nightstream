# Chip8EvidenceCoverage Spec

## Purpose

- **What it is**: The theorem-facing bridge from authenticated kernel proof
  objects to the semantic facts consumed by the CHIP-8 composition theorems.
- **Key property**: `semanticBounds_of_authenticatedEvidence`: if the final
  kernel proof exposes the exact public-input bindings, opening manifests,
  Stage-1 semantic objects, Stage-2 memory objects, Stage-3 continuity/bridge
  objects, and lower-layer opening refinements, then the semantic bounds
  consumed by `Chip8StepComposition` are derivable.
- **Protocol role**: This is the layer that closes the gap between the final
  kernel proof object and the row-local semantic execution theorems. It owns
  authenticated extraction and row-local seed export, not the chunk-global
  temporal closure objects consumed later by strong kernel soundness.

## Target Formulas

### Evidence inputs

This layer reasons over:

- authenticated public inputs and `meta_pub`
- `KernelOpeningManifest` and `RootOpeningManifest`
- lower-layer PCS opening witnesses together with the refinement map from those
  witnesses to raw scalar opening claims
- Stage-1 authenticated fetch/decode/ALU/Eq4/handoff objects
- Stage-2 authenticated register/RAM/Twist/RAF objects
- Stage-3 authenticated `LaneShiftProof`, continuity object, and row-binding
  claims
- authenticated row/view objects used to bind opened scalars back to the
  semantic lane row

### Object-level provenance

Top-level opening-manifest conformance alone is not sufficient. This module
introduces theorem-facing provenance predicates for the internal objects
consumed by checked relations.

$$
\mathrm{TableProvenance}(table)
$$

states:

- for every public Stage-1 table used by the kernel, the table object is backed
  by the exact committed table opening mandated by the kernel manifest;
- for evaluator-local helper semantics such as `Identity` and `Equal8`, a
  verifier-local evaluator may be used, but only as an explicitly separate
  local object rather than as a substitute for the committed public tables.

$$
\mathrm{VirtualValProvenance}(val, session)
$$

states that a register or RAM virtual `Val` object is authenticated exactly by
the matching Twist read/write/Val proof chain for that session.

$$
\mathrm{AddressProvenance}(dec, fam, addr)
$$

states that the address or key is the exact Stage-1 or Stage-2 family
projection fixed by `Chip8DecodeAddressBinding`.

$$
\mathrm{HandoffProvenance}(dec, handoff)
$$

states that the Stage-2 `usesY`, `readsRam`, and `writesRam` bits come from the
exact committed `C_decode_handoff` surface and equal the authenticated Stage-1
decode outputs.

### Twist session closure

The final kernel keeps the register and RAM Twist chains explicit. Introduce:

$$
\mathrm{TwistSessionClosed}(stage2)
$$

meaning:

- register read/write batching, `Val`-from-`Inc`, and address-correctness refer
  to one closed authenticated register session
- RAM read/write batching, `Val`-from-`Inc`, RAF support, and
  address-correctness refer to one closed authenticated RAM session
- no Stage-2 semantic fact is extracted from a dangling or mismatched subclaim
- every authenticated Twist read, write, and `Val` claim used by the row is
  covered by some session in the appropriate register-side or RAM-side
  registry
- the session key identifies one coherent read/write/`Val` triple rather than a
  multiset of conflicting local sessions

For this boundary, the session-key and registry-entry shape is fixed
positively:

- a register-side session key names one logical register timeline cell
  `(cycle_index, reg_addr)` over `V[0..15] ∪ {I} ∪ {⊥_reg}`
- a RAM-side session key names one logical RAM timeline cell
  `(cycle_index, ram_addr)` over `RAM[0..4095] ∪ {⊥_ram}`
- one registry entry packages exactly one read claim, one write claim, one
  `Val` claim, and the address/RAF provenance refs consumed later by row-local
  semantic extraction
- if two CHIP-8 roles on the same row touch the same logical cell, they must
  resolve to the same session key and therefore to the same registry entry
- sink-routed register roles are therefore part of the closed authenticated
  register registry rather than a side channel outside `TwistSessionClosed`

The exact theorem-facing register-side key surface is owned by
`Chip8RegisterSessionBoundary`; this module consumes that keying discipline as
part of the Stage-2 authenticated bundle rather than re-defining it ad hoc.

The Lean evidence layer therefore exposes separate register-side and RAM-side
session registries and their corresponding closure witnesses, rather than one
undifferentiated Stage-2 registry field. The main kernel Stage-2 proof surface
must expose those register-side and RAM-side registries and their closure
witnesses explicitly; this module does not treat them as implicit
implementation-side artifacts.

### Row/view consistency

The semantic layer still needs an explicit positive row-binding theorem. Define:

$$
\mathrm{RowProjectionWitness}(kernelManifest, rowClaims, \rho)
$$

$$
\mathrm{RowConsistent}(\rho, z, dec, pre, post)
$$

where \(\rho\) is tied to the same authenticated `C_lane` row-binding claims
that later feed `PreparedStep`, and carries only:

- the authenticated semantic row itself,
- verifier-known fixed coordinates,
- and row-local projections that are definitionally determined by that row and
  public constants.

`RowProjectionWitness` does **not** by itself justify Stage-1 decode objects,
Stage-2 virtual `Val` objects, closed Twist sessions, or Stage-3 boundary
transfer facts. Those facts must remain behind their own explicit provenance
predicates.

### PCS refinement and public-input authentication

Each directly opened scalar used by semantic extraction must carry an explicit
lower-layer refinement:

$$
\mathrm{OpeningRefinement}(\text{params}, \text{extract},
  \mathrm{RawScalarClaim}(b,p,v)).
$$

This layer also consumes:

$$
\mathrm{KernelPublicInputsBound}(publicInput, meta, romTable, init)
$$

from `Chip8KernelInputBinding`.

For every direct scalar consumed by this layer, accepted-opening provenance must
also enforce coordinatewise equality between the direct claim's
`claimedValues` and the exact-opening witness values in the same
`polynomialIds` order. Refinement binds one direct scalar claim to one accepted
exact opening; it does not replace that value-equality check.

The theorem-facing scalar carrier for that projected scalar view is
`AcceptedScalarOpening`, exported through the direct-value witness layer. It is
a scalar projection of the stronger direct-claim object
`AcceptedDirectOpening`, which keeps one direct manifest claim, one exact
lower-layer opening witness, and one refinement chain tied together at
vector level. Downstream Stage-1 / Stage-2 / Stage-3 consumers may not justify
a direct scalar by a bare refinement object alone or by a scalar projection
detached from that accepted direct-opening path.

### Evidence coverage of semantic facts

Define:

$$
\mathrm{SemanticEvidenceCovered}
(publicInput, meta, kernelManifest, rootManifest,
 stage1, stage2, stage3, romTable, init, pre, post, dec, z)
$$

to mean:

- `KernelPublicInputsBound(publicInput, meta, romTable, init)`
- `KernelOpeningBoundary(kernelManifest, rootManifest)`
- one explicit `Stage1AuthenticatedBundle`
- one explicit `Stage2AuthenticatedBundle`
- one explicit `Stage3AuthenticatedBundle`
- object-level provenance for every table, virtual `Val`, address family, and
  handoff object used by the current row
- a row/view witness tied to the authenticated `C_lane` row-binding claims
- lower-layer refinement, coordinatewise accepted-opening equality, and raw PCS
  opening separation for every direct scalar transitively consumed by the
  theorem-facing direct-opening bundles of the current row:
  - the row/view projection bundle
  - the Stage-3 current-row continuity bundle at `r_shift`
  - the Stage-3 start-boundary bundle at `j0_bits`
  - the Stage-3 final-boundary bundle at `j_last_bits`
- exact Stage-1, Stage-2, and Stage-3 CHIP-8 bindings for the current row

Define the explicit Stage-3 authenticated bundle:

$$
\mathrm{Stage3AuthenticatedBundle}(stage3, z)
$$

to package:

- `LaneShiftBound`
- one explicit `PaddedContinuityCheckBound`
- the refinement from that padded-domain check to `ContinuityBound`
- `StartBoundaryBound`
- `FinalBoundaryBound`
- `StartBoundaryMatches`
- `FinalBoundaryMatches`
- `RowBound`
- the direct-opening witness/refinement coverage for the current-row, start-row,
  and final-row scalar openings consumed by those Stage-3 obligations

Define the explicit Stage-1 authenticated bundle:

$$
\mathrm{Stage1AuthenticatedBundle}(\dots)
$$

to package:

- one authenticated row/view projection path for the current row
- one `FetchDecodeBound` witness bundle for fetch and decode
- one `LookupBound` witness bundle for the optional ALU / Eq4 lookup row facts
- the table-provenance and address-provenance objects required by those checked
  Shout claims

For the `simple` kernel boundary, the Stage-1 table-auth mode is frozen
channel-by-channel:

- fetch uses the exact committed `C_rom_table @ r_fetch_addr` opening;
- decode uses the exact committed `C_decode_table @ r_decode_addr` opening;
- ALU uses the exact committed `C_alu_table @ r_add8lo_addr` opening whenever
  the current row uses that subtable;
- Eq4 uses the exact committed `C_eq4_table @ r_eq4_addr` opening whenever the
  current row uses that table;
- verifier-local helper evaluators such as identity/equality shims remain
  explicit local helper objects, not `table_opening_claims`, and they never
  replace the committed public table surfaces.

This is the theorem-facing owner for Stage-1 checked-claim authentication. It
is distinct from PCS direct-opening refinement: the direct-opening/refinement
chain covers the row/view and Stage-3 direct scalar bundles, while Shout
lookup/read claims remain authenticated by their own checked-claim witnesses
plus provenance predicates.

Define the explicit Stage-2 authenticated bundle:

$$
\mathrm{Stage2AuthenticatedBundle}(\dots)
$$

to package:

- one `MemoryBound` witness bundle for the current row
- one closed authenticated register-side Twist session registry for the row
- one closed authenticated RAM-side Twist session registry for the row
- the address/value coverage consequences exported by those registries

This is the theorem-facing owner for Stage-2 checked-claim authentication. It
does not pretend that Twist read/write/`Val` checked claims are PCS direct
openings; instead it makes their soundness-carrying checked-claim path explicit.

For the current simple kernel, the Stage-2 initialization mode is fixed
positively, not left as an implicit implementation convention:

- `init_mode_id = authenticated_nonzero_init`
- register and RAM `Val` chains use the authenticated initial surfaces directly
- the Stage-2 `Val` consequences are therefore interpreted through the modified
  non-zero-init identity imported from `Chip8WitnessMemoryBinding`, not through
  a synthetic preload-write reduction.

### Extraction theorems

The key extraction targets are:

$$
\mathrm{SemanticEvidenceCovered}(\dots)
\Longrightarrow
\mathrm{chip8RoutingSound}(z)
$$

$$
\mathrm{SemanticEvidenceCovered}(\dots)
\Longrightarrow
\mathrm{KernelPublicInputsBound}(publicInput, meta, romTable, init)
$$

$$
\mathrm{SemanticEvidenceCovered}(\dots)
\Longrightarrow
\mathrm{FetchDecodeBound}(romTable, pre.PC, dec)
$$

$$
\mathrm{SemanticEvidenceCovered}(\dots)
\Longrightarrow
\mathrm{LookupBound}(dec, pre, z)
$$

$$
\mathrm{SemanticEvidenceCovered}(\dots)
\Longrightarrow
\mathrm{WitnessBinds}(pre, post, dec, z)
$$

$$
\mathrm{SemanticEvidenceCovered}(\dots)
\Longrightarrow
\mathrm{MemoryBound}(pre, post, init, dec, z)
$$

Define the row-local Stage-2 temporal seed bundle:

$$
\mathrm{Stage2TemporalSeedBound}(stage2, pre, post, dec, z)
$$

to package:

- the exact local memory trace object `trace`
- `TraceMatches(pre, post, dec, trace)`
- the opcode-local memory-frame fact
- register-port correctness
- RAM-port correctness
- RAM RAF support
- the closed authenticated register Twist-session registry for the current row
- the closed authenticated RAM Twist-session registry for the current row
- the row-local register-side address-coverage consequences tying authenticated
  sessions to `regRaX`, `regRaY`, `regRaI`, and `regWa`
- the read/write/`Val` totality and key-uniqueness consequences exported by
  those closed registries
- the row-local load/store address-coverage consequences exported from the
  RAM-side registry

This bundle is still row-local. It does **not** yet claim the whole-trace
register or RAM timeline theorem; it exports the exact Stage-2 seeds that a
later temporal owner must compose across the authenticated trace into one
chunk-global Stage-2 temporal context shared by all rows of the semantic
prefix.

In particular, this module does **not** own:

$$
\mathrm{Stage2TemporalContextBound}(\mathrm{trace})
$$

or

$$
\mathrm{PcAdjacentBridge}(\mathrm{trace}).
$$

Those are separate theorem-level closure objects consumed later by
`Chip8TwistTemporalInstantiation`, `Chip8PcContinuityBridge`,
`Chip8TemporalConsistency`, and then `Chip8AuthenticatedTrace`.

$$
\mathrm{SemanticEvidenceCovered}(\dots)
\Longrightarrow
\mathrm{ContinuityRowBound}(stage3, z)
$$

$$
\mathrm{SemanticEvidenceCovered}(\dots)
\Longrightarrow
\mathrm{LaneShiftSourceOpeningAppearsInManifest}(kernelManifest)
$$

$$
\mathrm{SemanticEvidenceCovered}(\dots)
\Longrightarrow
\exists\, startRow,\ \mathrm{StartBoundaryBound}(startRow)
$$

$$
\mathrm{SemanticEvidenceCovered}(\dots)
\Longrightarrow
\exists\, finalRow,\ \mathrm{FinalBoundaryBound}(finalRow)
$$

$$
\mathrm{SemanticEvidenceCovered}(\dots) \land stepIdx = 0
\Longrightarrow
\mathrm{StartBoundaryFrame}(\langle dec, pre, post, z \rangle)
$$

$$
\mathrm{SemanticEvidenceCovered}(\dots) \land stepIdx + 1 = meta.semanticRows
\Longrightarrow
\mathrm{FinalBoundaryFrame}(\langle dec, pre, post, z \rangle)
$$

Bundled together:

$$
\mathrm{SemanticEvidenceCovered}(\dots)
\Longrightarrow
\Big(
\mathrm{KernelPublicInputsBound}
\land
\mathrm{FetchDecodeBound}
\land
\mathrm{LookupBound}
\land
\mathrm{WitnessBinds}
\land
\mathrm{MemoryBound}
\land
\mathrm{ContinuityRowBound}
\land
\mathrm{LaneShiftSourceOpeningAppearsInManifest}
\Big).
$$

This is the exact staged-evidence bridge consumed by the later trace-level
closure layer. It does **not** by itself upgrade one authenticated row into
`ExecutionCorrect`, but it does discharge the row-local routing fact needed by
that later closure layer.

To keep the theorem-facing surface exact to the final kernel proof object,
the exported boundary may hide any internal stage-local claim lists used by the
Lean implementation. Define:

$$
\mathrm{ExactSemanticEvidenceCovered}(\dots)
$$

to mean that there exist whatever internal stage-local witnesses are needed to
establish `SemanticEvidenceCovered`, without exposing those internal lists as
top-level theorem parameters.

## Paper Anchors

- **Sources**:
  - `./docs/soundness-specs/twist-and-shout-requirements.md`
  - `./crates/neo-fold-prototype/specs/chip8-kernel.md`
- Anchors:
  - commitment-before-challenge discipline
  - Stage-1 linkage and handoff
  - Stage-2 explicit Twist chains
  - Stage-3 continuity and row binding
  - prohibition on treating sparse openings as unauthenticated full-row access

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Chip8/Stage2/EvidenceCoverage.lean` | Kernel-proof-to-semantics extraction theorems for the final CHIP-8 kernel |
| `Nightstream/Chip8/Stage2/EvidenceCoverageInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Provenance | `TableProvenance` | def | Definitional | Public Stage-1 tables are commitment-backed; evaluator-local helper tables remain explicitly separate |
| Provenance | `VirtualValProvenance` | def | Definitional | Every register/RAM virtual `Val` object is justified by the exact Stage-2 chain |
| Provenance | `AddressProvenance` | def | Definitional | Every Stage-1/Stage-2 address comes from the exact CHIP-8 family projection |
| Provenance | `HandoffProvenance` | def | Definitional | Every Stage-2 handoff bit comes from `C_decode_handoff` and equals the Stage-1 decode output |
| Stage 1 | `Stage1AuthenticatedBundle` | def/structure | Definitional | Packages the exact Stage-1 checked-claim surface, including the frozen per-channel table-auth mode of the simple kernel |
| Closure | `TwistSessionClosed` | def | Definitional | Stage 2 contains closed register and RAM Twist sessions |
| Closure | `RegisterTwistSessionRegistry` | abbrev | Definitional | Named register-side Twist registry surface |
| Closure | `RamTwistSessionRegistry` | abbrev | Definitional | Named RAM-side Twist registry surface |
| Closure | `RegisterTwistSessionClosed` | abbrev | Definitional | The register-side Twist registry is closed |
| Closure | `RamTwistSessionClosed` | abbrev | Definitional | The RAM-side Twist registry is closed |
| Rows | `RowProjectionWitness` | def | Definitional | Explicit authenticated row/local-view witness tied to `C_lane` row-binding claims only |
| Rows | `RowConsistent` | def | Definitional | The semantic row/view witness matches the consumed row objects |
| Inputs | `PCSContext` | def | Definitional | Lower-layer PCS parameters and scalar extractor for raw opening refinement |
| Coverage | `SemanticEvidenceCovered` | def | Definitional | Authenticated kernel evidence is sufficient to recover the semantic row facts |
| Coverage | `ExactSemanticEvidenceCovered` | def | Definitional | Final-kernel theorem-facing coverage predicate that hides internal stage-local witness lists |
| Theorem | `routingSound_of_evidence` | theorem | Theorem-Target | Authenticated evidence yields the exact row-local routing soundness `chip8RoutingSound(z)` |
| Theorem | `kernelPublicInputsBound_of_evidence` | theorem | Theorem-Target | Authenticated evidence yields the fixed public kernel input bounds |
| Theorem | `fetchDecodeBound_of_evidence` | theorem | Theorem-Target | Authenticated evidence yields Stage-1 fetch/decode binding |
| Theorem | `lookupBound_of_evidence` | theorem | Theorem-Target | Authenticated evidence yields Stage-1 helper-lookup semantics |
| Theorem | `witnessBinds_of_evidence` | theorem | Theorem-Target | Authenticated evidence yields semantic lane-row binding |
| Theorem | `memoryBound_of_evidence` | theorem | Theorem-Target | Authenticated evidence yields Stage-2 semantic memory binding |
| Coverage | `Stage2TemporalSeedBound` | structure | Definitional | Packages the exact row-local Stage-2 temporal seed objects consumed by later whole-trace owners |
| Extractor | `stage2TemporalSeedBound_of_evidence` | def | Noncomputable Extractor | Authenticated evidence yields the exact row-local Stage-2 temporal seed bundle |
| Extractor | `stage2TemporalSeedBound_of_exactAuthenticatedEvidence` | def | Noncomputable Extractor | Exact final-kernel evidence yields the same Stage-2 temporal seed bundle while hiding internal stage claim lists |
| Theorem | `continuityRowBound_of_evidence` | theorem | Theorem-Target | Authenticated evidence yields the Stage-3 row-local continuity facts needed downstream |
| Theorem | `startBoundaryBound_of_evidence` | theorem | Theorem-Target | Authenticated evidence yields the Stage-3 start-boundary row opening |
| Theorem | `startBoundaryFrame_of_evidence` | theorem | Theorem-Target | If `stepIdx = 0`, authenticated evidence yields the semantic-row start-boundary fact for the current frame |
| Theorem | `finalBoundaryBound_of_evidence` | theorem | Theorem-Target | Authenticated evidence yields the Stage-3 final-boundary row opening |
| Theorem | `finalBoundaryFrame_of_evidence` | theorem | Theorem-Target | If `stepIdx + 1 = meta.semanticRows`, authenticated evidence yields the semantic-row final-boundary fact for the current frame |
| Theorem | `semanticBounds_of_authenticatedEvidence` | theorem | Theorem-Target | Authenticated evidence yields the full semantic fact bundle consumed by `Chip8StepComposition` |
| Theorem | `semanticBounds_of_exactAuthenticatedEvidence` | theorem | Theorem-Target | Exact final-kernel evidence yields the same semantic fact bundle without exposing internal stage claim lists |

## Proof Obligations

- The module must track the final kernel proof object structure, not the older
  abstract Stage-1/2/3 claim-multiset abstraction.
- Any internal stage-local claim lists used to witness extraction must remain
  hidden behind an exact theorem-facing coverage predicate.
- The module must keep Stage-2 Twist closure explicit.
- The row/view witness must be tied to explicit `C_lane` row-binding claims.
- The module must not overclaim row-local staged evidence as if it already
  discharged `wf(z)` or trace start-boundary hypotheses that belong to later
  owners. Row-local routing soundness is part of the exact staged-evidence
  surface and must be exported here.
- Exporting authenticated Stage-3 boundary rows is required. This module must
  also export the row-local transfer facts that turn those openings into
  `StartBoundaryFrame` / `FinalBoundaryFrame` when the current row index is the
  start or final semantic index. Proving that one exact chunk really begins and
  ends at those indices remains a later trace-level owner.
- Exporting the row-local Stage-2 temporal seeds is required. This module must
  make the exact local trace/port/RAF/session objects available to later
  whole-trace temporal owners without pretending that row-local evidence alone
  already proves a trace-wide register or RAM timeline.
- Every direct scalar used in extraction must be backed by an explicit
  lower-layer opening refinement.
- No semantic fact may be extracted from a kernel proof object without
  explicit provenance.

## Assumption Ledger

- Generic Shout and Twist theorem statements are imported.
- PCS binding and Fiat-Shamir security remain external.
- This module owns only the CHIP-8 instantiation-level extraction story.

## Dependency and Consumer Map

- **Upstream dependencies**:
  - `Nightstream/Chip8/Kernel/OpeningBoundary.lean`
  - `Nightstream/Chip8/Stage1/FetchDecodeBinding.lean`
  - `Nightstream/Chip8/Stage1/DecodeAddressBinding.lean`
  - `Nightstream/Chip8/Stage2/WitnessMemoryBinding.lean`
  - `Nightstream/Chip8/Kernel/RomScheduleBinding.lean`
  - `Nightstream/Chip8/Stage3/ContinuityBridge.lean`
  - `Nightstream/PCSOpeningSemantics.lean`
- **Downstream consumers**:
  - `Nightstream/Chip8/Kernel/StagedExecutionDigest.lean`
  - `Nightstream/Chip8/Kernel/ArtifactAudit.lean`
  - `Nightstream/Chip8/Execution/StepComposition.lean`
  - later Rust-refinement theorems about the final kernel proof object

## Implementation Plan

1. Define provenance, closure, and row-binding predicates over the final kernel
   proof object.
2. Define `SemanticEvidenceCovered`.
3. Prove the per-fact extraction theorems.
4. Prove the bundled semantic-coverage theorem.

## Quality Expectations

- Keep the module focused on authenticated extraction.
- Keep the final kernel's Stage-1, Stage-2, and Stage-3 ownership split
  explicit.
- Keep row-binding and PCS refinement explicit and separate.
- Do not blur row-local Stage-2 seeds into the chunk-global
  `Stage2TemporalContextBound`.
- Do not claim the Stage-3 semantic `pc` bridge; that belongs to
  `Chip8PcContinuityBridge`.

## Acceptance Criteria

1. `lake build Nightstream.Chip8.EvidenceCoverage` succeeds.
2. The theorem surface rules out unauthenticated extraction from internal proof
   objects.
3. The theorem surface rules out treating sparse openings as free full-row
   access.
4. No `sorry`.

## Out of Scope

- generic Shout theorem proofs
- generic Twist theorem proofs
- the root main-lane CCS proof itself
- final PCS opening verification

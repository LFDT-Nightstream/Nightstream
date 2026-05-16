# Chip8TranscriptSchedule Spec

## Purpose

- **What it is**: The theorem-facing owner for the exact `root0` commitment
  bundle and the exact Fiat-Shamir transcript schedule of the 3-stage CHIP-8
  kernel.
- **Key property**: `challenge_after_phase0`: every stage challenge appears only
  after the exact phase-0 absorb of the kernel commitment bundle and
  `meta_pub`.
- **Protocol role**: This is the protocol-time owner required by the
  Twist/Shout commitment-before-challenge discipline. It does not own opening
  manifests or semantic extraction; it owns the exact order in which the kernel
  fixes commitments, records terminal points, samples challenges, and emits
  row-binding events. It stops at the final manifest-emission event; later
  accepted-opening verification, refinements, and any optional reduction
  artifacts are post-transcript materialization steps, not additional transcript
  events of this owner.

## Target Formulas

### Root0 commitment bundle

Define the exact phase-0 kernel commitment list:

$$
\mathrm{root0CommitmentIds}
:=
[
  C_{\mathrm{lane}},
  C_{\mathrm{fetch\_ra}},
  C_{\mathrm{decode\_ra}},
  C_{\mathrm{alu\_ra}},
  C_{\mathrm{eq4\_ra}},
  C_{\mathrm{decode\_handoff}},
  C_{\mathrm{reg}},
  C_{\mathrm{ram}},
  C_{\mathrm{rom\_table}},
  C_{\mathrm{decode\_table}},
  C_{\mathrm{alu\_table}},
  C_{\mathrm{eq4\_table}}
].
$$

Define the theorem-facing bound inventory of actual absorbed commitment
payloads:

$$
\mathrm{Root0CommitmentBinding} := (\mathrm{id}, \mathrm{digest}),
$$

and the conformance predicate:

$$
\mathrm{root0CommitmentBindingsConform}(bindings)
\;:=\;
\mathrm{map}\ \mathrm{id}\ bindings = \mathrm{root0CommitmentIds}.
$$

The phase-0 absorb events are:

$$
\mathrm{phase0Events}(bindings)
:=
\mathrm{map}\ \mathrm{absorbCommitment}\ bindings
\mathbin{+\!\!+}
[\mathrm{absorbMetaPub}].
$$

This is the theorem-facing `root0` boundary: all later challenges are sampled
 only after these events have occurred, and the absorbed commitment payloads are
 required to conform to `root0CommitmentIds`.

Here `root0` is a transcript-digest boundary over the absorbed kernel
commitment payloads plus `meta_pub`. It is not itself a commitment family, and
it is not a root-side opening surface.

Here `absorbMetaPub` is not an implementation-defined struct hash. It means the
exact labeled absorb sequence fixed by the kernel spec:

- absorb `protocol_version_id` under the root0 version label
- absorb `field_id`
- absorb `extension_field_id`
- absorb `program_image_digest`
- absorb `initial_state_digest`
- absorb `rom_table_digest`
- absorb `decode_table_digest`
- absorb `alu_table_digest`
- absorb `eq4_table_digest`
- absorb `transcript_seed_digest`
- absorb `root_params_id`
- absorb the ordered numeric suffix
  `(variable_order_id, domain_shape_id, sink_convention_id, init_mode_id,
  lowering_convention_id, padding_convention_id, table_auth_mode_id,
  opening_reduction_mode_id, program_word_count, semantic_rows,
  padded_trace_length, pad_pc_word, program_base_addr, cycle_bits)`

in exactly that order and under the exact root0 labels fixed by the kernel
spec.

This schedule is not only an abstract ordering law. It is also the canonical
protocol-binding absorb order that any executable Rust↔Lean compatibility lane
must reuse when deriving transcript-bound Poseidon2 digest and challenge
values.

Post-transcript verifier artifacts such as exact-opening verification records,
`OpeningRefinement` objects, and any future claim-space reduction summaries are
outside `KernelTranscriptSchedule`. This owner fixes the challenge-bearing
kernel schedule only.

### Canonical stage schedule

Define the exact canonical transcript events:

$$
\mathrm{stage1Events}
$$

containing, in order:

- sample shared Stage-1 cycle point `r_lookup`
- fetch / decode / ALU / Eq4 sumchecks
- address-correctness checks in canonical order
  `fetch -> decode -> alu -> eq4`
- record `r_fetch_addr`, `r_decode_addr`, `r_alu_addr`
- derive `r_add8lo_addr` from `r_alu_addr`
- record `r_eq4_addr`
- sample `γ_lookup_link`
- perform the Stage-1 linkage batch

Define:

$$
\mathrm{stage2Events}
$$

containing, in order:

- sample shared Stage-2 cycle point `r_twist_cycle`
- sample `γ_reg`
- register read/write batch
- register `Val`-from-`Inc`
- sample `γ_ram`
- RAM read/write batch
- RAM `Val`-from-`Inc`
- RAM RAF read/write support checks
- address-correctness checks in canonical order
  `RegRaX -> RegRaY -> RegRaI -> RegWa -> RamRa -> RamWa`
- record `r_addr_reg` and `r_addr_ram`
- sample `γ_twist_link`
- perform the Stage-2 linkage batch

Define:

$$
\mathrm{stage3PrefixEvents}
$$

containing, in order:

- sample `β1`, `β2`
- sample shared Stage-3 cycle point `r_shift`
- verify the lane-shift reduction
- verify continuity at `r_shift`
- open `j0_bits`
- open `j_last_bits`

and:

$$
\mathrm{stage3RowBindingEvents}(N)
:=
[\mathrm{rowBinding}(0), \dots, \mathrm{rowBinding}(N-1)].
$$

Then:

$$
\mathrm{transcriptEvents}(bindings, N)
:=
\mathrm{phase0Events}(bindings)
\mathbin{+\!\!+}
\mathrm{stage1Events}
\mathbin{+\!\!+}
\mathrm{stage2Events}
\mathbin{+\!\!+}
(\mathrm{stage3PrefixEvents} \mathbin{+\!\!+} \mathrm{stage3RowBindingEvents}(N))
\mathbin{+\!\!+}
[\mathrm{emitKernelOpeningClaims}].
$$

Define:

$$
\mathrm{KernelTranscriptSchedule}(bindings, N, events)
$$

to mean:

$$
\mathrm{root0CommitmentBindingsConform}(bindings)
\;\land\;
events = \mathrm{transcriptEvents}(bindings, N).
$$

### Challenge and terminal-point classes

Define:

$$
\mathrm{ChallengeEvent}(e)
$$

for the exact Fiat-Shamir challenge events:

- Stage-1 shared cycle point
- `γ_lookup_link`
- Stage-2 shared cycle point
- `γ_reg`
- `γ_ram`
- `γ_twist_link`
- `β1`
- `β2`
- Stage-3 shared cycle point

Define:

$$
\mathrm{Stage1TerminalPointEvent}(e)
$$

for:

- `recordFetchAddr`
- `recordDecodeAddr`
- `recordAluAddr`
- `recordEq4Addr`

Define:

$$
\mathrm{Stage2TerminalPointEvent}(e)
$$

for:

- `recordRegAddr`
- `recordRamAddr`

### Challenge prerequisite table

For auditor use, the exact local prerequisite schedule is:

- `r_lookup`
  - must already be bound: the actual `root0` commitment digests/encodings,
    absorbed in canonical `root0CommitmentIds` order via
    `root0CommitmentBindingsConform`, and then the exact labeled `meta_pub`
    absorb sequence
  - must not yet depend on: any Stage-1 proof transcript, any terminal address
    point, any later stage event
- `γ_lookup_link`
  - must already be bound: `r_lookup`, all Stage-1 Shout transcript events, and
    the recorded Stage-1 terminal points
  - must not yet depend on: any Stage-2 or Stage-3 event
- `r_twist_cycle`
  - must already be bound: the complete Stage-1 transcript including
    `γ_lookup_link`
  - must not yet depend on: any Stage-2 terminal point or later stage event
- `γ_reg`, `γ_ram`, `γ_twist_link`
  - must already be bound: `r_twist_cycle` and the preceding Stage-2 transcript
    events in exact local order
  - must not yet depend on: any Stage-3 event
- `β1`, `β2`, `r_shift`
  - must already be bound: the complete Stage-1 and Stage-2 transcripts
  - must not yet depend on: any Stage-3 row-binding or later post-transcript
    reduction artifact
- claim-local mixers, group-local mixers, and unification mixers
  - if a future non-simple format materializes them, they belong to dedicated
    post-transcript reduction domains rather than to `KernelTranscriptSchedule`
  - they must not reuse: `r_lookup`, any Stage-1 address point,
    `γ_lookup_link`, `r_twist_cycle`, `r_addr_reg`, `r_addr_ram`, `γ_reg`,
    `γ_ram`, `γ_twist_link`, `β1`, `β2`, or `r_shift`

### Theorem targets

The transcript owner must expose:

$$
\mathrm{root0CommitmentIds} \text{ has no duplicates}
$$

and:

$$
cid \in \mathrm{root0CommitmentIds}
\Longleftrightarrow
\mathrm{isKernelCommitment}(cid).
$$

This is the bridge back to `Chip8OpeningBoundary`: every kernel-manifest
commitment is one of the commitments fixed in `root0`.

The exact bridge theorem is:

$$
\mathrm{ExactKernelOpeningBoundary}(pts, kernelManifest, rootManifest)
\land
claim \in kernelManifest
\Longrightarrow
claim.commitmentId \in \mathrm{root0CommitmentIds}.
$$

The schedule owner must also prove:

$$
\mathrm{KernelTranscriptSchedule}(bindings, N, events)
\Longrightarrow
\exists rest,\ events = \mathrm{phase0Events}(bindings) \mathbin{+\!\!+} rest.
$$

and the commitment-before-challenge theorem:

$$
\mathrm{KernelTranscriptSchedule}(bindings, N, events)
\land
\mathrm{ChallengeEvent}(e)
\Longrightarrow
\exists rest,\ events = \mathrm{phase0Events}(bindings) \mathbin{+\!\!+} rest
\land e \in rest.
$$

Likewise, Stage-1 and Stage-2 terminal points must be proved to occur only in
the post-`root0` suffix.

The Stage-1 projection rule must also remain explicit:

$$
\neg \mathrm{ChallengeEvent}(\mathrm{deriveAdd8LoAddr}).
$$

This expresses that `r_add8lo_addr` is a deterministic projection of
`r_alu_addr`, not an independent transcript challenge.

Finally, Stage 3 row binding coverage must be exact:

$$
\mathrm{rowBinding}(j) \in \mathrm{stage3RowBindingEvents}(N)
\Longleftrightarrow
j < N.
$$

and therefore:

$$
\mathrm{KernelTranscriptSchedule}(N, events)
\Longrightarrow
(\mathrm{rowBinding}(j) \in events \Longleftrightarrow j < N).
$$

The final emit step must also be last:

$$
\mathrm{KernelTranscriptSchedule}(N, events)
\Longrightarrow
\exists prefix,\ events = prefix \mathbin{+\!\!+}
[\mathrm{emitKernelOpeningClaims}].
$$

## Paper Anchors

- **Sources**:
  - `./crates/neo-fold-prototype/specs/chip8-kernel.md`
  - `./docs/soundness-specs/twist-and-shout-requirements.md`
  - `./docs/assurance-strategy.md`
- Anchors:
  - exact `root0` commitment bundle
  - commitment-before-challenge discipline
  - exact Fiat-Shamir ordering for the 3-stage kernel
  - Stage-1 / Stage-2 terminal-point recording rules
  - Stage-3 row-binding coverage over the active semantic prefix

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Chip8/TranscriptSchedule.lean` | Exact `root0` commitment bundle and transcript-order theorems |
| `Nightstream/Chip8/TranscriptScheduleInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Events | `TranscriptEvent` | def | Definitional | Canonical theorem-facing transcript event alphabet |
| Phase 0 | `root0CommitmentIds` | def | Definitional | Exact kernel commitments absorbed into `root0` |
| Phase 0 | `Root0CommitmentBinding` | def | Definitional | Exact `(commitment id, absorbed digest/encoding)` inventory entry |
| Phase 0 | `root0CommitmentBindingsConform` | def | Definitional | Binds the absorbed root0 payload inventory to canonical `root0CommitmentIds` order |
| Phase 0 | `phase0Events` | def | Definitional | Exact phase-0 absorb prefix over actual root0 commitment payloads |
| Stages | `stage1Events` | def | Definitional | Exact Stage-1 transcript segment |
| Stages | `stage2Events` | def | Definitional | Exact Stage-2 transcript segment |
| Stages | `stage3PrefixEvents` | def | Definitional | Exact Stage-3 pre-row-binding segment |
| Stages | `stage3RowBindingEvents` | def | Definitional | Exact row-binding event list over `[0, N)` |
| Bundle | `transcriptEvents` | def | Definitional | Full canonical transcript schedule |
| Bundle | `KernelTranscriptSchedule` | def | Definitional | Exact transcript equality predicate |
| Classes | `ChallengeEvent` | def | Definitional | Exact challenge-event classifier |
| Classes | `Stage1TerminalPointEvent` | def | Definitional | Exact Stage-1 terminal-point classifier |
| Classes | `Stage2TerminalPointEvent` | def | Definitional | Exact Stage-2 terminal-point classifier |
| Theorem | `root0CommitmentIds_nodup` | theorem | Theorem-Target | Root0 commitment inventory is duplicate-free |
| Theorem | `root0CommitmentBindings_ids` | theorem | Theorem-Target | A conforming absorbed root0 inventory projects back to canonical `root0CommitmentIds` |
| Theorem | `mem_root0CommitmentIds_iff_isKernelCommitment` | theorem | Theorem-Target | Root0 commitments are exactly the kernel commitments |
| Theorem | `kernelClaim_commitment_fixed_in_root0` | theorem | Theorem-Target | Every conforming kernel-manifest claim references a commitment fixed in `root0` |
| Theorem | `challenge_after_phase0` | theorem | Theorem-Target | Every challenge event occurs after the exact phase-0 absorb prefix |
| Theorem | `stage1TerminalPoint_after_phase0` | theorem | Theorem-Target | Every Stage-1 terminal-point event occurs after phase 0 |
| Theorem | `stage2TerminalPoint_after_phase0` | theorem | Theorem-Target | Every Stage-2 terminal-point event occurs after phase 0 |
| Theorem | `deriveAdd8LoAddr_not_challenge` | theorem | Theorem-Target | `r_add8lo_addr` is derived, not sampled |
| Theorem | `rowBinding_event_in_schedule_iff` | theorem | Theorem-Target | Stage-3 row-binding events cover exactly the active semantic rows |
| Theorem | `emitKernelOpeningClaims_last` | theorem | Theorem-Target | Kernel opening claims are emitted only after all prior transcript events |

## Proof Obligations

- This owner must stay protocol-time only.
- It must not re-own manifest shape; that remains in `Chip8OpeningBoundary`.
- It must make the commitment-before-challenge discipline theorem-facing.
- It must keep `r_add8lo_addr` explicit as a deterministic projection.
- It must keep Stage-3 row-binding coverage exact over `[0, N)`.

## Assumption Ledger

- This module does not prove PCS security or Fiat-Shamir security.
- It proves the exact transcript-order contract required by the instantiated
  kernel, not the cryptographic soundness of the transcript primitive itself.
- It does not prove Stage-1 / Stage-2 / Stage-3 semantics.

## Dependency and Consumer Map

- **Upstream dependencies**:
  - `Nightstream/Chip8/OpeningBoundary.lean`
- **Downstream consumers**:
  - later protocol-level kernel soundness owners
  - later artifact-audit owners that need an exact transcript-order contract
  - reviewers auditing the instantiated Twist/Shout commitment boundary

## Acceptance Criteria

1. `lake build Nightstream.Chip8.TranscriptSchedule` succeeds.
2. The theorem surface explicitly owns `root0` and the exact transcript order.
3. No `sorry`.

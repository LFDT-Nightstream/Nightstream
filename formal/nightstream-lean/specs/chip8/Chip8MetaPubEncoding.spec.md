# Chip8MetaPubEncoding Spec

## Purpose

- **What it is**: The theorem-facing owner for the exact `KernelMetaPub`
  payload and the canonical labeled `meta_pub` absorption sequence used at
  `root0`.
- **Key property**: `root0MetaPubAbsorbPlan` fixes one exact ordered list of
  labeled absorb operations for `meta_pub`; it is not an implementation-defined
  struct serialization.
- **Protocol role**: This owner fixes the exact public-metadata payload that the
  transcript layer must consume before any later challenge is sampled.

## Target Formulas

### Canonical public metadata object

Define one theorem-facing public metadata object:

$$
\mathrm{KernelMetaPub}
$$

with fields:

- `program_image_digest`
- `initial_state_digest`
- `rom_table_digest`
- `decode_table_digest`
- `alu_table_digest`
- `eq4_table_digest`
- `transcript_seed_digest`
- `protocol_version_id`
- `field_id`
- `extension_field_id`
- `root_params_id`
- `variable_order_id`
- `domain_shape_id`
- `sink_convention_id`
- `init_mode_id`
- `lowering_convention_id`
- `padding_convention_id`
- `table_auth_mode_id`
- `opening_reduction_mode_id`
- `program_word_count`
- `semantic_rows`
- `padded_trace_length`
- `pad_pc_word`
- `program_base_addr`
- `cycle_bits`

### Relation-shaping mode identifiers

For the current `simple` kernel boundary, the relation-shaping mode fields are
not free-form labels. They are fixed positively to the exact theorem surfaces
already owned elsewhere in this bundle:

- `init_mode_id = authenticated_nonzero_init`
- `lowering_convention_id = chip8_microstep_pre_post_v1`
- `table_auth_mode_id = committed_public_tables_v1`
- `opening_reduction_mode_id = no_post_transcript_reduction_v1`

with exact meaning:

- `authenticated_nonzero_init` means the Stage-2 register and RAM `Val` chains
  use the authenticated initial surfaces directly, via the modified non-zero-init
  identity from `Chip8WitnessMemoryBinding`;
- `chip8_microstep_pre_post_v1` means the exact row-granular lowering and
  same-row visibility order from `chip8-kernel.md` §3.4, including separate
  register/RAM timelines and the explicit `Fx55/Fx65` burst decomposition;
- `committed_public_tables_v1` means fetch/decode/ALU/Eq4 table authentication
  is carried by the exact committed public-table openings fixed by the simple
  kernel boundary; verifier-local helper evaluators remain cross-checks only;
- `no_post_transcript_reduction_v1` means the simple boundary exports no
  post-transcript claim-space reduction summaries and no family-local fold
  carriers.

Any other value for one of these fields is non-conforming under the current
`protocol_version_id`.

### Public-kernel projection

Define:

$$
\mathrm{kernelMetaCore}(metaPub)
$$

to project the shared public-input fields already owned by
`Chip8RomScheduleBinding`.

### Ordered numeric suffix

Define:

$$
\mathrm{metaPubNumericSuffix}(metaPub)
$$

to be the exact ordered list:

$$
[
  variable\_order\_id,
  domain\_shape\_id,
  sink\_convention\_id,
  init\_mode\_id,
  lowering\_convention\_id,
  padding\_convention\_id,
  table\_auth\_mode\_id,
  opening\_reduction\_mode\_id,
  program\_word\_count,
  semantic\_rows,
  padded\_trace\_length,
  pad\_pc\_word,
  program\_base\_addr,
  cycle\_bits
].
$$

### Canonical labeled absorb plan

Define one absorb-operation type:

$$
\mathrm{Root0MetaAbsorbOp}
$$

and define:

$$
\mathrm{root0MetaPubAbsorbPlan}(metaPub)
$$

to be the exact ordered absorb list:

1. `("chip8/root0/version", [protocol_version_id])`
2. `("chip8/root0/field_id", [field_id])`
3. `("chip8/root0/extension_field_id", [extension_field_id])`
4. `("chip8/root0/program_image_digest", program_image_digest)`
5. `("chip8/root0/initial_state_digest", initial_state_digest)`
6. `("chip8/root0/rom_table_digest", rom_table_digest)`
7. `("chip8/root0/decode_table_digest", decode_table_digest)`
8. `("chip8/root0/alu_table_digest", alu_table_digest)`
9. `("chip8/root0/eq4_table_digest", eq4_table_digest)`
10. `("chip8/root0/transcript_seed_digest", transcript_seed_digest)`
11. `("chip8/root0/root_params_id", root_params_id)`
12. `("chip8/root0/meta_pub", metaPubNumericSuffix(metaPub))`

This owner fixes the exact payload and label order. It does not yet apply the
Poseidon2 permutation.

## Paper Anchors

- **Sources**:
  - `./docs/assurance-strategy.md`
  - `./crates/neo-fold-prototype/specs/chip8-kernel.md`
  - `./formal/nightstream-lean/specs/chip8/Chip8RomScheduleBinding.spec.md`
  - `./formal/nightstream-lean/specs/chip8/Chip8TranscriptSchedule.spec.md`

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Chip8/Kernel/MetaPubEncoding.lean` | Exact `KernelMetaPub` payload and canonical labeled absorb plan |
| `Nightstream/Chip8/Kernel/MetaPubEncodingInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Structure | `KernelMetaPub` | structure | Definitional | Exact theorem-facing public metadata payload for `root0` |
| Projection | `kernelMetaCore` | def | Definitional | Projects the shared public-input core |
| Encoding | `metaPubNumericSuffix` | def | Definitional | Exact ordered numeric suffix |
| Structure | `Root0MetaAbsorbOp` | inductive | Definitional | One labeled absorb operation in the `meta_pub` root0 payload |
| Encoding | `root0MetaPubAbsorbPlan` | def | Definitional | Exact labeled absorb plan for `meta_pub` |
| Theorem | `metaPubNumericSuffix_length` | theorem | Theorem-Target | The canonical numeric suffix has length `14` |
| Theorem | `root0MetaPubAbsorbPlan_length` | theorem | Theorem-Target | The canonical absorb plan has length `12` |
| Theorem | `root0MetaPubAbsorbPlan_labels` | theorem | Theorem-Target | The absorb plan uses the exact canonical label order |

## Proof Obligations

- The theorem surface must make the exact `meta_pub` payload explicit.
- The canonical absorb plan must be label-exact and order-exact.
- This owner must not collapse `meta_pub` into an implementation-defined struct
  hash.
- This owner stays below the actual transcript permutation; it owns payload
  order, not Poseidon2 itself.

## Acceptance Criteria

1. `lake build Nightstream.Chip8.Kernel.MetaPubEncoding` succeeds.
2. The exact `meta_pub` payload is Lean-owned.
3. The exact labeled absorb order is Lean-owned.
4. No `sorry`.

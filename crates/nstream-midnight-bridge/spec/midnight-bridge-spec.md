# nstream-midnight-bridge Specification

## 1. Scope

`nstream-midnight-bridge` owns the final Midnight proof backend for theorem-facing
RV64IM Nightstream exports produced by `neo-fold-prototype`.

Its job is:

- take the frozen public Nightstream boundary exported by
  `crates/neo-fold-prototype/src/nightstream/mod.rs` and
  `crates/neo-fold-prototype/src/nightstream/rv64im.rs`,
- encode the verifier for that boundary as a Midnight circuit,
- lower that circuit to a form accepted by unmodified `external/midnight-ledger`,
- and emit Midnight prover/verifier material plus a final Midnight proof.

It does not own:

- the RV64IM arithmetization,
- the SuperNeo backend contract,
- the inner folding relation,
- the recursive proof construction inside `neo-fold-prototype`,
- the public Nightstream theorem boundary itself,
- or the current proof-complete RV64IM transport object as a final theorem input.

## 2. Fixed External Constraints

The bridge is constrained by the current Midnight stack.

1. `external/midnight-ledger` is treated as immutable.
2. The proof server accepts serialized ZKIR sources today:
   - `zkir::IrSource` (`v2`) by default,
   - `zkir_v3::IrSource` only behind Midnight's experimental feature.
3. Midnight proving uses its native proving system and verifier-key model.
4. Midnight prover parameters and verifier parameters are KZG-based:
   - `ParamsKZG<Bls12>`
   - `ParamsVerifierKZG<Bls12>`
5. Therefore the bridge must target Midnight-native proving, not a foreign final
   proof system.

Consequences:

- `neo-fold-prototype` should stop at the public Nightstream boundary.
- `nstream-midnight-bridge` is the outer compression backend.
- The bridge must lower its verifier relation to accepted ZKIR.
- The bridge must not require ledger/runtime/proof-server patches.
- Any wider RV64IM proof material still required today belongs to the
  bridge-private witness, not to the public theorem input.

## 3. Bridge Input From `neo-fold-prototype`

The bridge input is the theorem-facing public Nightstream boundary, not the
legacy proof-complete RV64IM transport object.

For RV64IM v1, the public theorem-facing bridge input is:

- `NightstreamStatement`
- and the Midnight chain-owned binding input if the contract path requires one

The first Midnight prototype should expose only a fixed public image of that
statement:

- `bridge_version`
- `nightstream_statement_digest`
- and the Midnight chain-owned binding input if the contract path requires one

The bridge may additionally consume a **bridge-private witness** sufficient to
justify that public statement under the current verifier relation.

Until a narrower bridge-private carrier is frozen, that private witness should
include:

- canonical `NightstreamStatement` encoding,
- backend-specific compact `Rv64imNightstreamProof`,
- bridge-private proof-binding and linkage consistency claims derived from the
  verified RV64IM seam,
- and current backend-accounted RV64IM proof material sufficient to derive and
  justify those claims

That wider witness may be derived from or include material such as:

- `Rv64imProof`
- `Rv64imAcceptedProofArtifact`
- verified final-statement / final-proof seam material
- separate backend-accounted Spartan proof material

Those wider objects are permitted only as bridge-private witness. They are not
final theorem inputs and must not widen the public Nightstream boundary.

The bridge does not consume as final theorem inputs:

- stage package digests,
- kernel opening digests,
- root lane columns,
- raw trace rows,
- witness-side duplication,
- separate backend-accounted proof material,
- or any other proof-complete audit sidecars.

## 4. Neo-Side Ground Truth

The bridge spec must match the actual Neo relation being exported.

### 4.1 Root RV64IM relation

The live RV64IM main lane is one uniform root relation, not per-opcode custom
R1CS.

Current owned facts:

- root semantic row width: `38`
- root public inputs: `1`
- root relation shape: fixed `29`-row R1CS embedding carried as CCS

The root relation is owned by:

- `crates/neo-fold-prototype/src/rv64im/ccs.rs`
- `crates/neo-fold-prototype/specs/riscv-kernel.md`

The “58” value in RV64IM parity code is not the root constraint count.

### 4.2 SuperNeo proof spine

The live proof spine is:

```text
Π_CCS -> Π_RLC -> Π_DEC
```

Both prover and verifier replay that sequence with Poseidon2 on the Nightstream
side.

`Π_CCS` already includes:

- FE sumcheck,
- NC sumcheck,
- public challenges,
- final running sums,
- and `header_digest`.

So the exported theorem being bridged is the full public-coin SuperNeo reduction
path over the fixed RV64IM root CCS, not just “29 equations per step.”

### 4.3 Current RV64IM public proof surfaces

The current RV64IM public Nightstream surface is the compact carried boundary in:

- `crates/neo-fold-prototype/src/nightstream/mod.rs`
- `crates/neo-fold-prototype/src/nightstream/rv64im.rs`

That public surface is already theorem-facing and compact.

The current `Rv64imProof` export remains proof-complete transport. Its public
digests, packaged proofs, and witness duplication are transitional bridge/audit
surfaces only.

The bridge must therefore treat:

- `NightstreamStatement` as the public theorem input,
- `Rv64imNightstreamProof` as backend-specific private witness material,
- and `Rv64imProof` plus any accepted/final seam expansions as bridge-private
  implementation material unless a narrower carrier is frozen explicitly.

## 5. Bridge Theorem

The bridge proves:

> the carried RV64IM Nightstream boundary exported by `neo-fold-prototype`
> verifies under the current Nightstream verifier relation, therefore the
> claimed RV64IM execution statement holds.

Operationally:

- `neo-fold-prototype` owns the inner folding relation and the compact Nightstream
  public boundary,
- `nstream-midnight-bridge` owns only the outer Midnight verifier circuit for
  that boundary,
- the final portable proof is the Midnight proof emitted by this crate.

The bridge must not re-prove the full RV64IM execution relation directly.
Its circuit is verifier-sized, not execution-sized.

If the current Nightstream verifier still requires wider backend-accounted proof
material to justify the compact carried boundary, that material shall enter only
as bridge-private witness.

## 6. Required Midnight Output

The bridge must emit artifacts consumable by unmodified Midnight tooling:

- `ProvingKeyMaterial`
  - `prover_key`
  - `verifier_key`
  - `ir_source`
- `ProofPreimage`
- Midnight `ProofVersioned`
- Midnight `VerifierKey` suitable for contract installation

The default compatibility target is proof-server and contract usage through the
existing Midnight interfaces, not a custom side channel.

For chain-facing install paths, the bridge may mirror Midnight verifier-key
wrapper wire formats locally when doing so avoids importing larger ledger-side
assembly machinery into this crate. The public Nightstream theorem boundary
still remains unchanged.

For RV64IM v1, the bridge may also own the typed operands of verifier-key
installation, such as the entry-point wrapper and the versioned verifier-key
insert object, so deploy/update callers do not pass raw tagged bytes around.

For the same RV64IM v1 contract path, the bridge may also own a typed unsigned
verifier-key insert-update payload carrying the contract address, maintenance
counter, and insert operand. This is a bridge-owned client/install helper, not a
claim that `nstream-midnight-bridge` owns the full signed Midnight maintenance
transaction format.

That RV64IM v1 unsigned install/update helper may still own the exact Midnight
`data_to_sign` preimage for the verifier-key insert path, so external signers
can sign the canonical maintenance bytes without reconstructing tagged storage
state outside the bridge namespace.

For the same RV64IM v1 path, the bridge may also own canonical signature
attachment and exact signed maintenance-update bytes for that verifier-key
insert object, as long as this remains a chain/install helper and not a new
public theorem-facing Nightstream surface.

For that same signed RV64IM v1 install path, the bridge may also own the exact
reverse parse from `contract-maintenance-update[v1]` bytes back into the typed
verifier-key insert update object, so deploy/update callers do not interpret
chain-facing maintenance bytes outside this namespace.

For the same RV64IM v1 contract path, the bridge may also own the exact
`contract-action[v6]` `Maintain` wrapper over that signed verifier-key insert
update, so submission callers do not assemble or decode chain-facing contract
action bytes outside this namespace.

If the local client path wants a submission helper, it should remain narrow:
the bridge may own a typed submit-request wrapper over those exact action bytes
plus transport error normalization, while leaving chain response semantics as
opaque receipt bytes until a real external submission contract is frozen.

That narrow submit helper may accept either the exact `Maintain` action object
or the signed verifier-key insert update directly, as long as the bridge
internally owns the `contract-action[v6]` wrapping step and emits the same
exact action bytes in both cases.

## 7. ZKIR Target

The bridge circuit must be emitted as ZKIR.

Default target:

- `zkir::IrSource` (`v2`)

Optional later target:

- `zkir_v3::IrSource`, but only as a separately versioned compatibility mode

`v2` is the default because it is accepted by Midnight proof-server without the
experimental feature.

The bridge must not depend on a custom serialized relation type.

## 8. Circuit Boundary

The Midnight circuit owns only the verification of the compact RV64IM
Nightstream boundary using bridge-private witness material as needed.

It must verify, at minimum:

- the private `NightstreamStatement` encoding is well-formed,
- the private `Rv64imNightstreamProof` encoding is well-formed,
- the public `nightstream_statement_digest` matches the private statement,
- the Nightstream verifier relation accepts for that private boundary,
- the verified statement matches the private statement,
- bridge-private proof-binding claims are internally consistent and match the
  carried private statement / proof fields,
- bridge-private linkage claims are internally consistent and match the carried
  private statement / proof fields,
- and any required Midnight chain binding input is bound.

It must not:

- reconstruct RV64IM execution rows,
- widen the public theorem input with proof-complete current bridge sidecars,
- or translate the full Neo root CCS into Midnight constraints.

The correct boundary is:

```text
Nightstream public boundary + bridge-private witness -> Midnight circuit -> Midnight proof
```

not:

```text
Neo RV64IM execution relation -> Midnight circuit
```

### 8.1 First Prototype Boundary

The first bridge prototype shall mirror the current exported RV64IM Nightstream
verifier relation in:

- `crates/neo-fold-prototype/src/nightstream/rv64im.rs`
  - `verify_rv64im_nightstream(...)`
  - plus the bridge-owned `root_params_id -> verifier_context_digest` compatibility check

not a separate chunk-verifier prototype over `PublicChunk` and `ChunkProof`.

For the current Rust reference / ZKIR-v2 bring-up, the first verifier-shaped
IR is allowed to check only explicit bridge-owned consistency relations that do
not change the Nightstream hash theorem. In particular, it should check:

- `bridge_version == 1`
- public `nightstream_statement_digest` words equal a bridge-private statement
  digest witness
- private `NightstreamStatement.proof_binding_root` equals a bridge-private
  proof-binding root witness
- bridge-private proof-binding claim digests equal the corresponding carried
  private `Rv64imNightstreamProof` digests
- private `NightstreamStatement.verifier_context_digest` equals a
  bridge-private verifier-context digest witness derived from the carried
  `root_params_id`
- private `NightstreamStatement.fold_schedule` equals a bridge-private
  fold-schedule witness derived from the verified RV64IM final seam
- private `NightstreamStatement.public_io_digest` equals private
  `Rv64imMainResidualProof.public_statement_digest`
- private `Rv64imMainResidualProof.kernel_export_proof_digest` equals the
  bridge-private linkage anchor digest
- private `NightstreamStatement.linkage_root` equals a bridge-private
  linkage-root witness
- private `NightstreamStatement.chunk_summaries[*].public_chunk_digest` equals
  the bridge-private linkage public-chunk digests
- private `NightstreamStatement.chunk_summaries[*].chunk_relation_digest`
  equals the bridge-private chunk-transition claimed relation digests
- private `Rv64imMainResidualProof.chunk_transition_digests[*]` equals the
  bridge-private chunk-transition witness digests
- private `NightstreamStatement.chunk_summaries[*].start_index` forms a
  contiguous partition from zero under the carried `public_step_count` values
- private `NightstreamStatement.chunk_summaries.len()` equals private
  `Rv64imMainResidualProof.chunk_transition_digests.len()`
- private `NightstreamStatement.semantic_step_count` equals the sum of private
  `chunk_summaries[*].public_step_count`
- and all transcript lanes are fully consumed by the IR

It must not pretend to recompute the Poseidon2 Nightstream digests with a
different hash family. Exact hash-faithful recomputation is a later closure
step, not something to fake inside the first prototype.

Specifically, the first bridge unit should take as **public** input:

- `bridge_version`
- `nightstream_statement_digest`
- Midnight chain binding input if required

and as **private** witness:

- canonical `NightstreamStatement` encoding,
- backend-specific compact `Rv64imNightstreamProof`,
- a bridge-private carrier sufficient to reconstruct the accepted artifact and
  verified final seam required by the current Nightstream verifier relation,
- any auxiliary arithmetic witness needed by Midnight gadgets

The initial Midnight circuit should therefore mirror only these checks:

1. `NightstreamStatement` decoding plus explicit equality against a
   bridge-private `statement_digest_hint`,
2. bridge-private verifier-context and fold-schedule consistency with the
   carried statement,
3. bridge-private proof-binding-claims consistency with the carried statement
   and carried compact proof,
4. main residual/public statement consistency,
5. bridge-private linkage-root and linkage-claims consistency with the carried
   statement and linkage artifact,
6. bridge-private chunk-transition binding consistency with the carried
   statement and carried residual proof,
7. contiguous fixed-shape chunk layout consistency inside the carried
   statement,
8. main residual/kernel-export consistency with the bridge-private linkage
   anchor digest,
9. chunk-count / semantic-step-count consistency across the carried private
   boundary,
10. full transcript-lane consumption,
11. and required chain binding-input consistency.
    For the local RV64IM v1 prototype, this means `binding_input` equals the
    first canonical field-word of `nightstream_statement_digest`, enforced by
    the bridge preimage verifier rather than the first IR.

It shall not mirror in the first prototype:

- raw RV64IM execution rows,
- the full root CCS directly,
- or any proof-complete transport field as public theorem input.

## 9. Public and Private Data Mapping

This crate must define a stable mapping from the Nightstream public boundary
plus bridge-private witness into
Midnight's `ProofPreimage`.

Required roles:

- `inputs`
  - `bridge_version`
  - `nightstream_statement_digest`
  - plus the chain-owned binding input if required
- `private_transcript`
  - bridge-private consistency claims
  - canonical `NightstreamStatement` encoding
  - canonical backend-specific compact proof encoding
  - serialized wider bridge-private witness bytes
  - plus any bridge-private auxiliary witness needed by the ZKIR verifier
- `public_transcript_inputs`
  - empty unless the chosen ZKIR verifier architecture requires them
- `public_transcript_outputs`
  - empty unless the chosen ZKIR verifier architecture requires them
- `communications_commitment`
  - `None` unless the chosen ZKIR verifier architecture requires it
- `binding_input`
  - Midnight chain-owned binding input as supplied by the existing contract path
  - for the local RV64IM v1 prototype, it must equal the first canonical
    field-word of `nightstream_statement_digest`
  - any serialized proof-server request must carry this value inside the
    embedded `ProofPreimage`
  - the bridge-owned `/prove` request helper must leave the optional
    binding-input override slot empty
  - the bridge-owned `/check` request helper must serialize only
    `(proof-preimage-versioned, Option<wrapped-ir>)`, with no extra binding
    override channel
  - bridge-owned `/check` and `/prove` helpers should take bridge-owned request
    policy objects that choose resolver-backed vs embedded material, not raw
    `Option` knobs or caller-managed key-lookup policy at the outer API
    boundary
  - bridge-owned client adapters should then map those policies into typed
    `/prove` and `/check` requests so callers do not assemble route strings and
    serialized bodies by hand
  - bridge-owned client adapters should also parse `/prove` and `/check`
    responses back into typed bridge-owned response objects, so callers do not
    decode `proof-versioned` or check-output vectors outside this namespace
- `key_location`
  - stable versioned namespace for bridge keys
  - for the RV64IM v1 bridge, it must equal
    `nstream-midnight-bridge/rv64im/nightstream/v1`

This mapping must be versioned explicitly. A change in public Nightstream
encoding, bridge-private witness encoding, or chain binding layout must change
the bridge version.

### 9.1 First Prototype Mapping

For the first RV64IM prototype, the split should be:

- `inputs`
  - bridge version,
  - fixed field-vector encoding of `nightstream_statement_digest`,
  - no additional public replay of chain binding input; the local RV64IM v1
    prototype derives `binding_input` from the first canonical field-word of
    `nightstream_statement_digest`
- `private_transcript`
  - fixed field-vector encoding of bridge-private consistency claims
    - statement digest hint
    - proof-binding consistency claims
    - linkage consistency claims
  - fixed field-vector encoding of `NightstreamStatement`
  - fixed field-vector encoding of `Rv64imNightstreamProof`
  - fixed or versioned bytes-to-field encoding of the wider bridge-private witness
  - auxiliary arithmetic witness needed by Goldilocks and `K` gadgets

The first prototype should avoid exposing any wider proof-complete transport
structure as public input. If the bridge-private witness initially contains
`Rv64imProof` or `Rv64imAcceptedProofArtifact`, the Midnight circuit should
derive bridge-private consistency claims from that wider witness natively and
constrain the carried private `NightstreamStatement` / `Rv64imNightstreamProof`
to match those claims inside the IR. Exact Poseidon2 hash-faithful digest
recomputation of `proof_binding_root` and `linkage_root` is a later closure
step, not part of the first prototype.

## 10. Hashing Boundary

This spec draws a hard boundary between Nightstream-owned hashes and
Midnight-owned backend internals.

Nightstream-owned hashes inside the bridged verifier relation must remain
Poseidon2:

- statement digests,
- transcript replay for the Nightstream recursive/export verifier,
- any public accumulator handle digests,
- any bridge-owned digest packing inside the verified Nightstream artifact.

Midnight-owned proving internals are external backend details:

- Midnight prover transcript,
- Midnight verifier transcript,
- Midnight chain binding-input derivation.

`nstream-midnight-bridge` must not introduce any new bridge-local Blake2b or
SHA-256 prehashing on top of the Nightstream artifact. It may only rely on the
existing Midnight-native mechanisms already required by the chain.

## 11. Non-Goals

This crate does not do any of the following:

- convert the full Neo RV64IM R1CS/CCS relation into ZKIR,
- define a foreign final proof system and then wrap it inside Midnight,
- own the current `Rv64imProof` transport format as the final theorem,
- preserve current auxiliary digests as permanent theorem inputs by default,
- or depend on `tmp/removed-neo-midnight-bridge` as a protocol template.

The removed bridge is reference material only for:

- Goldilocks embedding tricks,
- digest packing patterns,
- and circuit sizing intuition.

## 12. Implementation Plan

| Step | Work | Output |
|---|---|---|
| 1 | Implement a minimal Nightstream-boundary reference unit over `nightstream_statement_digest` plus private `NightstreamStatement` + `Rv64imNightstreamProof` + bridge-private witness. | First bridge unit with fixed public/private split. |
| 2 | Define canonical field encoding for that unit's public digest and private witness vectors. | Stable prototype input layout. |
| 3 | Implement the current RV64IM Nightstream verifier relation in Rust first as a typed reference verifier over the encoded payload. | Reference verifier for circuit cross-checks. |
| 4 | Lower that verifier relation to ZKIR v2. | `IrSource` generator or checked-in IR artifact. |
| 5 | Generate `ProvingKeyMaterial` for the prototype bridge circuit. | Midnight-native prover key, verifier key, and IR source. |
| 6 | Prove locally through Midnight proof-server using the prototype bridge preimage. | Passing `/check` and `/prove` smoke tests. |
| 7 | Shrink the bridge-private witness toward a narrower verifier-owned carrier without changing the public Nightstream boundary. | Versioned final bridge input contract. |
| 8 | Install the verifier key on a contract entry point and prove a real contract path on local Midnight infrastructure. | End-to-end contract verification. |
| 9 | Add size and cost measurements for bridge proof generation and verification. | Feasibility gate for chain use. |

## 13. Acceptance Criteria

The bridge is complete only when all of the following are true:

1. `neo-fold-prototype` exports the compact public Nightstream boundary as the final
   theorem-facing public proof for RV64IM.
2. `nstream-midnight-bridge` converts that boundary plus any required
   bridge-private witness into Midnight
   `ProvingKeyMaterial` and `ProofPreimage`.
3. Midnight proof-server can prove and check the bridge circuit without any
   ledger modifications.
4. A verifier key for the bridge circuit can be installed for a contract entry
   point.
5. A contract path can accept the resulting Midnight proof.
6. The bridged public statement remains the compact Nightstream theorem boundary,
   not a replay of current Neo proof-complete transport digests.

## 14. Immediate Next Work

The next implementation tasks are:

1. define the first bridge unit over public `nightstream_statement_digest`
   plus private `NightstreamStatement`, `Rv64imNightstreamProof`, and bridge-private witness,
2. choose the field-vector encoding for that public digest and private witness inside Midnight,
3. build a typed reference verifier for the current RV64IM Nightstream relation in
   `nstream-midnight-bridge`,
4. add a first ZKIR smoke test proving that verifier-shaped prototype through
   Midnight proof-server,
5. then freeze the narrowest bridge-private witness carrier that still proves
   the compact public Nightstream boundary without widening it.

# Current circuit correspondence

Status: **selected target; G4 evidence open**.

The circuit target is `PaddedRowIdentity`. The current legacy Split-NC
circuit is not evidence for this target unless a new proof connects the exact
relations.

## Required correspondence claims

One current circuit manifest must support four independent claims:

1. **Completeness:** each normative verifier acceptance has a satisfying
   circuit assignment for the same statement and proof.
2. **Soundness:** each arbitrary satisfying assignment implies normative
   verifier acceptance, or a named cryptographic bad event.
3. **Public-image binding:** each circuit public vector decodes to the exact
   nine-field `public_image_v1` tuple. The circuit recomputes the digest from
   the selected verifier-key digest and full canonical statement field stream.
4. **Lowering:** hints, lookups, ranges, frontend rows, backend rows, and the
   verifier key preserve the same relation.

Honest witness-generation tests support only part of the first claim.

## Exact circuit target

The circuit must implement:

- the typed v1 statement and proof field layout after canonical outer decoding;
- the selected verifier-key digest as a fixed circuit constant;
- the verifier-key-owned 24-variable structure;
- the artifact-owned `M_0=[I_m;0]` and 13 padded application matrices;
- one 24-round degree-9 joint PiCCS SumCheck;
- 15-by-14 ordered terminal ring evaluations;
- the M0-bound norm terminal;
- 15 bounded PiRLC ring challenges;
- 14 deterministic PiDEC children;
- every selected Poseidon2 duplex step that determines a challenge or the
  public statement digest;
- rejection for sampler exhaustion and all malformed inputs;
- the canonical nine-field terminal public image and exact statement prehash.

The circuit manifest and lowering proof must show that the fixed verifier-key
digest equals the native recomputation from the canonical setup and Structure.
The circuit must not parse or hash the full sparse Structure only to repeat
that fixed-key check. The deployed parser remains responsible for canonical
container bytes.

The post-challenge PiRLC-output, PiDEC-output, and fold-finalization frames feed
only the verifier-derived fold receipt. They do not affect acceptance or the
public image. A circuit can omit those diagnostic hash rows. If it exposes the
receipt, its manifest must map and prove the extra rows separately.

It must not contain a relation-authoritative column terminal, `y_zcol`,
`s_col`, `y_carrier`, extra beta challenge, or legacy proof fallback.

## Current manifest

The manifest must bind:

1. contract and profile hashes;
2. source revision, build identity, and feature set;
3. proof variant and Structure identity;
4. transcript, sampler, codec, and `public_image_v1` identities;
5. frontend and backend relation hashes;
6. verifier-key and public-input identities;
7. exact row-family counts and row digest;
8. one rule owner for every non-plumbing row family;
9. every hint and its validating constraints;
10. exact theorem names for all four correspondence claims;
11. Rust-origin positive and adversarial evidence hashes;
12. terminal decider manifest identity.

The manifest parser must reject an unknown field, duplicate owner, missing row
owner, unsupported proof variant, stale hash, or profile mismatch.

## Required implementation order

1. Make the native verifier implement the selected protocol.
2. Implement the exact `public_image_v1` tuple and statement prehash.
3. map each verifier decision to one circuit row family;
4. prove arbitrary-assignment soundness;
5. prove honest completeness;
6. prove public-image decoding and statement-digest binding;
7. prove hint and frontend-to-backend lowering;
8. prove that the shipping compiler emits the checked manifest;
9. connect the manifest to the selected decider and deployed parser.

G4 stays open until all four claims and the current-manifest claim close for
one exact build. Local fixtures and historical circuit proofs do not close it.

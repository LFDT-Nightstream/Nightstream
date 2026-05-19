# Rv64imIvc

## Purpose

- **What it is**: The RV64IM-native incremental-verifiable-computation carrier and
  its optional terminal compression boundary.
- **Paper anchors**:
  - HyperNova Construction 2 / Definition 12
  - HyperNova §6.2 NIFS-compatible multi-folding
  - SuperNeo §7 `Π_CCS -> Π_RLC -> Π_DEC`
- **Scope**:
  - native base-case initialization
  - native append across repeated folds
  - native verification of the carried IVC state
  - optional terminal compression into one SNARK

## Contract Surface

The module exposes two artifacts:

| Artifact | Role |
|---|---|
| `Rv64imIvcState` | Serializable native carrier for repeated folds with no Spartan involvement |
| `Rv64imIvcSnark` | Optional compressed artifact derived from a finished native carrier |

## Native State Contract

`Rv64imIvcState` satisfies all of the following:

1. **Canonical base case**:
   `init` uses the canonical default pair `(u_perp, w_perp)` determined by the
   current `enc_str(F')` context.
2. **Serializable and resumable**:
   A state can be serialized, stored, deserialized, and used as the owner of
   later append operations.
3. **Native append**:
   `append` advances the carrier using the native Construction-2 / NIFS step
   only. The append boundary does not invoke Spartan.
4. **Native verify**:
   `verify` checks the carried native state and public image without invoking
   Spartan.
5. **Version binding**:
   The state carries the `vk_fs` digest so a change to `F'` / `enc_str(F')`
   invalidates previously serialized states.

## Compression Contract

`compress` converts a valid `Rv64imIvcState` into `Rv64imIvcSnark`.

Compression satisfies:

1. Spartan is used only at this boundary.
2. The compressed artifact binds one authoritative public image.
3. `Rv64imIvcSnark.verify` requires only:
   - the SNARK verifier key
   - the authoritative public image

## Invariants

- Native transcript and public-binding paths are Poseidon2-only.
- Native append and native verify do not depend on Spartan.
- The public image is authoritative only when recomputed from the carried native
  state or verified through the compressed boundary.

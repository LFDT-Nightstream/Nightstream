# Nightstream protocol-contract completion review

Status: **implementation-ready specification; production assurance open**.

`review.md` contains the detailed semantic review. This document records the
architecture and handoff result.

## Completed contract work

The package now has:

- 104 atomic normative requirements in small authored modules;
- one semantic requirement DAG with 170 direct edges;
- one exact 12-event fold-verifier state machine;
- four challenge families and five bounded repetitions;
- 20 approved Nightstream decisions with derived rule impact;
- exact Goldilocks, Phi81, Structure encoding, verifier-key digest, Poseidon2,
  transcript schedule, sampler, commitment, circuit-interface, decider-family,
  and threat-model profiles;
- separate Lean, Rust, Rust-origin, circuit, and security evidence ledgers;
- structured issues and a derived assurance DAG;
- review receipts bound to semantic, profile, and evidence identities;
- deterministic generated reading views and a complete manifest;
- package, repository-anchor, arithmetic, schema, and fault-injection checks.

## Selected protocol

The normative target is `PaddedRowIdentity`. It uses one padded row cube and
one reviewed joint PiCCS SumCheck. The first matrix is
`M_0=[I_11437038;0]`, so the norm terminal is an output evaluation of the same
witness that opens the commitment.

The selected profile removes the earlier rectangular FE/NC split, separate
column opening, carrier replay, and extra beta challenges. The public input
`x` is the only public-carrier authority.

## Maintenance result

Authored data and generated views are separate. Rule text, metadata, decisions,
protocol flow, profile values, evidence, and assurance claims each have one
owner. The checker derives reverse impact, status, freshness, gates, and
release flags. It does not assert that the current package must be blocked.

The checker rejects:

- duplicate IDs or clauses;
- mixed paper and decision authority;
- missing or duplicate evidence rows;
- requirement or claim cycles;
- redundant transitive graph edges;
- incomplete decision coverage;
- profile arithmetic or protocol-order drift;
- stale generated views or review receipts;
- source or package-manifest tampering.

## Handoff boundary

The contract is ready to be the implementation target when the derived status
shows G0, G0B, and G1 closed. Work should then proceed in this order:

1. implement and prove the padded-identity semantic refinement;
2. make the native Rust verifier conform to the exact event model;
3. produce independent Rust-origin positive and mutation evidence;
4. replace the legacy circuit and prove all four correspondence directions;
5. bind one concrete terminal manifest and deployed verifier;
6. evaluate the complete security reduction against the 96-bit policy.

## Claims that remain forbidden

Implementation readiness does not mean:

- the current Rust verifier conforms;
- the current circuit corresponds;
- the Lean padded-identity proof is complete;
- the Fiat-Shamir reduction is complete;
- the concrete decider or on-chain verifier is fixed;
- Nightstream has an end-to-end production security level.

These are G2 through G5 claims. They must remain open until current,
contract-bound evidence closes them.

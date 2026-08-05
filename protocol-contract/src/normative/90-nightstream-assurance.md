## 9. Conformance, circuit, and security evidence

### NS-RUST-EVIDENCE-ORIGIN — Rust execution origin

Rust conformance evidence MUST name the contract and profile hashes, Rust
revision, source-tree and lock hashes, feature set, compiler, target, producer
binary, command, and run attestation. The observed decisions and mutations
MUST originate in that Rust execution.

Decision: NSD-PROVENANCE-001.

### NS-RUST-EVIDENCE-CONTENT — Independent semantic check

Each evidence item MUST contain a rule-indexed ordered trace, complete input,
first rejection rule, and at least one adversarial mutation. An independent
checker MUST recompute the semantic result and validate every bound hash. A
carried acceptance Boolean MUST NOT establish conformance.

Decision: NSD-PROVENANCE-001.

### NS-CIRCUIT-MANIFEST — Current-circuit identity

The current shipping circuit MUST publish a manifest that binds the contract,
profile, transcript, sampler, source build, frontend relation, backend
relation, verifier key, public-input count, and exact rule-to-row map.
The manifest MUST identify the fixed circuit constant for the verifier-key
digest and the evidence that equates it to native recomputation from the
canonical setup and Structure.

Decision: NSD-CIRCUIT-001 and NSD-PROVENANCE-001.

### NS-CIRCUIT-COMPLETENESS — Native-to-circuit direction

For every normative verifier acceptance, the correspondence proof MUST show
that a satisfying circuit witness exists for the same statement and proof.

Decision: NSD-CIRCUIT-001.

### NS-CIRCUIT-SOUNDNESS — Circuit-to-native direction

For every arbitrary satisfying circuit assignment, the correspondence proof
MUST derive normative verifier acceptance or a named cryptographic bad event.
Honest witness-generation tests alone do not close this claim.

Decision: NSD-CIRCUIT-001.

### NS-CIRCUIT-PUBLIC-INPUT — Public-image decoder and statement binding

Every circuit public vector MUST contain the exact nine fields in
`public_image_v1.public_field_order`. The circuit MUST recompute its four digest
fields with the fresh-duplex session, verifier-key, statement, and squeeze
steps in that profile. The five explicit fields MUST match the selected profile
and decoded statement. A noncanonical public alias MUST be unsatisfiable or
reject. Two distinct preimages with one digest are a named Poseidon2 collision
event, not an encoding alias.

Decision: NSD-CIRCUIT-001, NSD-ENCODING-001, and NSD-HASH-001.

### NS-CIRCUIT-LOWERING — Hints and backend lowering

The circuit proof MUST cover hint constraints, lookups, ranges, frontend row
generation, and frontend-to-backend lowering. Every authoritative field MUST
have one owner and every non-plumbing row MUST map to one contract rule.

Decision: NSD-CIRCUIT-001.

### NS-DECIDER-CORRESPONDENCE — Terminal and deployed verifier

The terminal proof and deployed verifier MUST use the same backend manifest,
canonical public image, Poseidon2 transcript, parser, and verifier key. The
reduction MUST start at the deployed acceptance predicate, not a test fixture.

Decision: NSD-DECIDER-001, NSD-ENCODING-001, and NSD-TRANSCRIPT-001.

### NS-SEC-REDUCTION — Named bad-event ledger

The end-to-end theorem MUST bound separate terms for SumCheck, algebraic
mixing roots, padded-identity refinement, coordinate forking, relaxed binding,
strong extraction, Poseidon2 and Fiat-Shamir, sampler exhaustion, encoding,
implementation transfer, circuit soundness, backend proof, and deployed
verification. Each nonzero term MUST have a theorem, substitution, and owner.

Decision: NSD-SECURITY-001, NSD-THREAT-MODEL-001, and
NSD-REDUCTION-FRAMEWORK-001.

### NS-SEC-COMPOSITION — Lifetime and extractor composition

The reduction MUST compose the one-fold bound across at most 64 adaptive
folds and the stated oracle limit. If it uses a union bound, it MUST establish
a uniform conditional bound for every accepted prefix. The proof-of-knowledge
statement MUST give a concrete expected-polynomial-time extraction bound.

Decision: NSD-SECURITY-001 and NSD-THREAT-MODEL-001.

### NS-RELEASE-IMPLEMENTATION — Implementation-ready gate

Implementation work MAY target this contract only when G0, G0B, and G1 are
closed from current evidence. This state means the semantics and design are
fixed; it does not claim that Lean, Rust, circuit, or security evidence is
complete.

Decision: NSD-PROVENANCE-001.

### NS-RELEASE-PRODUCTION — Production release gate

A production release MUST close G2 through G5 and MUST NOT claim an assurance
tier above its weakest required edge. Any implementation difference from this
profile is a release blocker or a new versioned contract decision.

Decision: NSD-PROVENANCE-001, NSD-CIRCUIT-001, NSD-DECIDER-001, and
NSD-SECURITY-001.

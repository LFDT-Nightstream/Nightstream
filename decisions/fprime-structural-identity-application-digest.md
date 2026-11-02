# F′ Structural Identity With One Application Hash

**Status:** Accepted

## Problem

The structural identity of a per-application package hashed the sealed
envelope. The envelope holds the application rows and witness recipes two
times: in the package at index 1 and in the application plan at index 3. The
verifier context hashed the plan a third time for its application component.
Each hash is one sequential Poseidon2 sponge over all application rows, so the
compile time of a large application was mostly identity hashing.

Commit `620247d88` (2026-08-30) chose raw words for each binding: the
relation identifier, the application component of the verifier context, the
package identity and the verification key each hash raw words, and the
component digests are only recomputed compression. Each binding proof then
had one collision case.

## SuperNeo

SuperNeo v1.2 does not fix how an implementation identifies its relation or
its static key material. HyperNova Construction 2 needs only a verifier key
that binds the step relation.

## Decision

The owner selected this design on 2026-10-31. It changes the raw-words choice
of `620247d88` for the relation identifier only, because hashing the
application three times made the compile time mostly identity hashing.

1. The application digest is the verifier-context application component,
   `Lifecycle.VerifierContext.componentDigest 2` of the plan's words
   (`Export.Stage1.PerApplicationCanonicalPackage.applicationDigest`). It is
   the only hash of the application.
2. The structural identity hashes `structuralValue`: the sealed envelope with
   the application digest at index 3 and the fixed package at index 1. The
   fixed package is the package without the plan's witness batches, witness
   instructions and assertion rows
   (`Export.Stage1.PerApplicationCanonicalPackage.structuralValue`).
3. The digest is in the value one time. The fixed package and the plan give
   the package (`Export.Stage1.PerApplicationCanonicalPackage.reassemble_fixedPackage`),
   so a second copy of the digest in the package binds nothing more.
4. The verifier computes the application digest from the raw plan words
   (`ApplicationPackage.authorityWords`; Rust `ApplicationIdentity::from_words`
   and `native_application_identity`). It never takes the digest from the
   package or a prover. A digest stays compression, not authority.
5. The structural identity and the application digest keep separate jobs.
   The structural identity identifies the whole sealed package. The
   application digest is one of the four authority components of the
   verifier context (relation, application, NIFS key, commitment key).
6. The domain, `Nightstream/FPrime/package/v2`, and the emitted package format
   do not change. Each identity, verifier-context digest and key that uses
   the structural identity changes.

Poseidon2 stays the only hash. Equal structural identities give equal sealed
envelopes, two structural values with one hash, or two application plans with
one application digest
(`Export.Stage1.PerApplicationSecurity.structuralPackageIdentity_identifies_sealedPackage_or_collision`).
The last event is the new collision case of the binding proofs. It is a
collision of the application component, which the verifier context already
names
(`Layout.Stage1.PiCCSSecurity.AuthorityComponentDigestCollision`). See
[SPEC.md](../SPEC.md), section 4.4.

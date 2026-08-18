## 8. Nightstream verifier profile

### NS-AUTH-STRUCTURE — Verifier-key authority

The fold verifier key MUST own the selected profile ID, exact Structure, and
Ajtai setup seed and dimensions. A proof-supplied copy is non-authoritative
and MUST match or reject. The terminal backend manifest has separate authority
under NS-DECIDER-PROFILE.

The canonical relation artifact MUST use format
`nightstream/verifier-key-relation`, schema 1, and payload encoding
`rust-ccs-structure-serde-json-v1`. It MUST contain the complete thirteen
application matrices and polynomial. Its validator recomputes the expected
artifact from the live verifier-owned Structure and rejects unless the
complete decoded payload is equal. Matching dimensions or digests alone grant
no authority. This JSON artifact does not replace the canonical sparse
Structure stream used by NS-ENC-STRUCTURE for verifier-key hashing.

Decision: NSD-AUTHORITY-001 and NSD-ENCODING-001.

### NS-AUTH-CLAIM — CE claim authority

A CE claim MUST contain exactly `c`, `x`, `r`, and `y_ring` as defined by
SN-REL-CE. Shape values MUST come from the verifier key. An untyped or extra
relation field MUST reject.

Decision: NSD-AUTHORITY-001 and NSD-COLUMN-001.

### NS-AUTH-DERIVED — Caches and digests

The verifier MUST recompute constant-term caches, the verifier-key digest,
public-carrier evaluations, offsets, lengths, and the final fold
transcript digest from authoritative inputs. A digest or cache MUST NOT grant
authority to its preimage.

Decision: NSD-AUTHORITY-001, NSD-HASH-001, and NSD-PROVENANCE-001.

### NS-ENC-BASE — Canonical base-field bytes

A base-field element MUST be one 8-byte little-endian unsigned integer less
than `q`. Reduction modulo `q`, alternate integer widths, and noncanonical
values MUST reject.

Decision: NSD-ENCODING-001.

### NS-ENC-EXTENSION — Canonical extension encoding

`K_ext` MUST be `F_q[U]/(U^2-7)`. An element MUST encode as two canonical
base fields in `(c0,c1)` order for `c0+c1*U`.

Decision: NSD-ENCODING-001.

### NS-ENC-RING — Canonical ring encoding

An `R_F` or `R_K` value MUST use coefficients of degrees 0 through 53 in
that order. Each `R_K` coefficient MUST use NS-ENC-EXTENSION. No storage
padding or alternate coefficient order is valid.

Decision: NSD-ENCODING-001.

### NS-ENC-CONTAINER — Typed statement and proof containers

The statement and proof MUST use the v1 magic, version, variant, ordered
section IDs, and exact field and byte counts from the profile. Their decoders
use checked integer arithmetic and compare the complete input length before
payload allocation. Unknown, missing, duplicate, reordered, truncated, or
trailing content MUST reject. The statement is exactly 318,832 bytes and
39,848 base fields. The proof is exactly 463,528 bytes, with sections of 480,
22,680, and 34,776 base fields. One statement-proof pair MUST encode one fold.
A bounded sequence MUST contain exactly `fold_count` pairs in increasing
`fold_index` order.

Decision: NSD-ENCODING-001 and NSD-TRANSCRIPT-001.

### NS-ENC-STRUCTURE — Canonical Structure stream

The Structure MUST use `nightstream-sparse-structure-v1`. Its header and the
implicit `M_0` and zero-padding variants MUST use the profile layout. Each
application matrix MUST use strict row-major nonzero triples with valid indices,
nonzero canonical values, and no duplicates. `f_app` MUST use sorted unique
coefficient-and-exponent terms of degree 1 through 8, with maximum degree 8;
thus `f_app(0,...,0)=0`.

Decision: NSD-ENCODING-001 and NSD-AUTHORITY-001.

### NS-ENC-COMMITMENT — Commitment encoding

A commitment MUST encode as 18 `R_F` elements in row order under NS-ENC-RING.

Decision: NSD-ENCODING-001.

### NS-POSEIDON-PARAMETERS — Exact permutation

All protocol-binding Poseidon2 calls MUST use the values in
`src/profile/poseidon2-goldilocks-v1.toml`: Goldilocks width 8, rate 4,
capacity 4, `x^7`, 4 initial full rounds, 22 partial rounds, 4 terminal full
rounds, and the listed constants and matrices. The permutation MUST use the
listed column-vector operation order and match both profile test vectors.

Decision: NSD-HASH-001 and NSD-TRANSCRIPT-001.

### NS-TRANSCRIPT-SPONGE — Field duplex state transition

The transcript MUST start with eight zeros and absorb cursor 0. Absorption MUST
add a field to `state[cursor]`, increment the cursor, and permute and reset it
to 0 after lane 3. A frame MUST absorb `tag`, payload length, then payload. A
direction change MUST use the profile constants: add 1 at the cursor and 2 at
lane 3 before first squeeze, or add 3 at lane 4 and permute before absorption.

Decision: NSD-TRANSCRIPT-001 and NSD-HASH-001.

### NS-TRANSCRIPT-FRAMING — Squeeze and continuation rules

Before a squeeze, the transcript MUST frame the squeeze tag and requested lane
count under the challenge-frame tag. The first block MUST return rate lanes 0
through 3 in order after the direction permutation. Each continuation block
MUST add 4 at lane 4, permute, and return lanes 0 through 3. A caller MUST use
only the requested lane count.

Decision: NSD-TRANSCRIPT-001.

### NS-VERIFIER-KEY-DIGEST — Exact verifier-key prehash

The verifier MUST start a fresh selected field duplex, frame the contract domain
and profile version, then frame the setup code, dimensions, 32 seed-byte lanes,
and canonical Structure stream with their declared counts. It MUST squeeze four
base fields under the verifier-key-digest tag. The main fold transcript MUST
absorb those four fields under the verifier-key tag.

Decision: NSD-ENCODING-001, NSD-HASH-001, and NSD-TRANSCRIPT-001.

### NS-TRANSCRIPT-ORDER — Fold transcript schedule

The transcript order MUST be session, verifier-key digest, statement, PiCCS
input, alpha, gamma, 24 ordered SumCheck round messages and challenges, PiCCS
outputs, indexed PiRLC sampler attempts, derived PiRLC output, PiDEC children,
and fold finalization. Each fold in a bounded sequence MUST start with a fresh
zero-state duplex and use the sequence's single selected verifier key and
profile. The four-field fold transcript digest MUST be a verifier-derived
receipt and MUST NOT enter CE authority or the next fold statement. No
challenge may be sampled before all messages on which it depends are absorbed.

Decision: NSD-TRANSCRIPT-001, NSD-PICCS-001, and NSD-SAMPLER-001.

### NS-CHALLENGE-EXTENSION — Extension-field challenges

An `alpha`, `gamma`, or SumCheck challenge MUST be two consecutive uniform
base-field squeeze lanes interpreted as `(c0,c1)`. Zero is valid. Alpha MUST
contain 24 elements and SumCheck MUST sample one element after each round
message.

Decision: NSD-TRANSCRIPT-001 and NSD-BATCH-COINS-001.

### NS-SAMPLER-CANDIDATES — Uniform strong-set digits

For source index `i`, coefficient index `j`, and attempt `a`, the transcript
MUST absorb those indices under the PiRLC-candidate tag and squeeze one base
field. It MUST accept only `x<q-1` and map `x mod 5` in order to
`[-2,-1,0,1,2]`, then encode that signed digit with `iota_q` as coefficient
`j` of the source's ring challenge.

Decision: NSD-SAMPLER-001 and NSD-TRANSCRIPT-001.

### NS-SAMPLER-REPETITIONS — Bounded sampler loop

The verifier MUST process 17 sources in order, 54 coefficients per source,
and at most three attempts per coefficient. If all three candidates reject,
the whole proof MUST reject.

Decision: NSD-SAMPLER-001.

### NS-SAMPLER-LOSS — Sampler distribution and exhaustion

Accepted digits MUST be exactly uniform on the five-element alphabet. The
security reduction MUST include a per-fold exhaustion bound of at most
`918/q^3` and MUST compose it across the selected fold limit.

Decision: NSD-SAMPLER-001 and NSD-SECURITY-001.

### NS-SECURITY-POLICY — V1 threat and resource limits

The v1 security target MUST be at least 96 classical bits for one proof and
one session per verifier key with at most 64 folds. The resource census MUST
allow at most 262,144 adaptive oracle queries, including the derived maximum
178,049 prescribed tagged squeezes per key. The release theorem MUST be an
expected-polynomial-time proof of knowledge and MUST state the Ajtai setup or
seeded-PRG assumption.

Decision: NSD-SECURITY-001 and NSD-THREAT-MODEL-001.

### NS-DECIDER-PROFILE — Terminal backend interface

The terminal family MUST be `Spartan-with-WHIR-profile-manifest-v1` over the
selected current-circuit R1CS and the nine-field `public_image_v1` tuple. The
verifier key MUST own one versioned concrete backend manifest. An absent,
unknown, or mismatched manifest MUST reject.

Decision: NSD-DECIDER-001, NSD-CIRCUIT-001, NSD-ENCODING-001, and NSD-HASH-001.

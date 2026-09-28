# SuperNeo to Nightstream v1 semantic review

Review result: **pass for G0B semantic normalization and G1 design use**.

This result means that the contract gives one coherent implementation target
with traceable authority. It does not prove the Lean theorems, Rust refinement,
circuit correspondence, terminal proof, or end-to-end security reduction.

The bound assurance receipts identify this as a Codex-assisted protocol review.
They do not claim an external human audit.

## 1. Review scope

The review covered:

- the locked reviewed SuperNeo paper files;
- exact reverse and forward application of reviewed errata v4;
- every literal paper item in `src/paper/`;
- all 52 paper-derived atomic normative rules;
- all 52 Nightstream atomic normative rules;
- all 20 Nightstream decisions;
- the 170-edge semantic requirement DAG;
- the exact profile, state machine, transcript, sampler, and security planning
  arithmetic.

The review boundary was source meaning and contract normalization. Existing
Lean and Rust files were used only to find design conflicts and candidate
owners. Their presence was not treated as proof.

## 2. Method

For each paper-derived rule, the reviewer performed these checks:

1. identify the exact locked source item and reviewed errata, if any;
2. compare quantifiers, domains, bounds, indices, signs, targets, and losses;
3. confirm that the literal model does not add Nightstream behavior;
4. confirm that the normative paper rule cites only paper authority;
5. confirm that each Nightstream change cites only approved decisions;
6. inspect direct semantic dependencies and remove redundant transitive edges;
7. recompute each exact profile count from lower-level values;
8. check that a replacement gives complete final behavior.

The checker independently validates unique IDs, block hashes, authority type,
duplicate clauses, cycles, redundant edges, decision coverage, profile
arithmetic, protocol order, schema validity, and generated-view freshness.

## 3. Paper normalization results

### Foundations and relations

The contract preserves:

- `n_F=d*n_R` and `n_F,in=d*n_R,in` as logical paper dimensions;
- centered coefficient norms and strict `Bound_a(z)` semantics;
- `B=b^k<q/2` as the protocol bound;
- `B_amb=floor(q/2)+1` as the ambient extraction bound;
- consecutive ring coefficient embedding and the full `R_F` module action;
- verifier ownership of one Structure;
- the corrected CCS zero set and corrected `L_in` domain;
- the exact `CCS` and `CE` relation fields.

No Nightstream padding fact enters these paper rules. Padding appears only in
Nightstream additions.

### Reduction framework

The contract now includes the source material that the earlier draft omitted:

- the interactive reduction interface;
- reduction-of-knowledge requirements;
- sequential composition;
- strong and weak reductions for one shared projection;
- PiCCS strong condition (i), which fixes the output commitment image across
  independent prover runs;
- PiRLC extractor agreement for that same projection;
- the exact shared-point ambient relation `BatchCE_15(B_amb,L)`.

The fold proof uses strong-weak composition for `PiRLC o PiCCS`, then sequential
composition with PiDEC. PiDEC is not inserted into the strong-weak theorem.
Definition 19 and the folding-scheme type are traced to `SRC-PAPER-12`; they are
not attributed only to the reduction chapter.

### SumCheck and PiCCS

The normalized rules include all reviewed corrections:

- total degree for `D_f` and individual degree for SumCheck;
- `D_Q=max(D_f+1,2b,2)`;
- strict norm roots;
- the absolute shifted target `T_abs`;
- corrected source, matrix, coefficient, and gamma indexing, including the map
  from carried local index `c` to global source `K_fresh+c`;
- separate SumCheck and independent root-event terms;
- output `y_(i,j)` for each source and matrix;
- Lemma 7 as an atomic if-and-only-if rule that uses the same ordered source
  vectors for CCS truth, strict norm, and carried evaluations;
- identity-output binding of the norm terminal;
- an unconditioned first extractor run;
- success-gated retries and the global `sqrt(delta)` disagreement term;
- ambient extraction into `BatchCE_15(B_amb,L)`.

### PiRLC, PiDEC, and commitment

The normalized rules keep:

- one ring challenge per PiCCS output;
- the full ring-module action in every mixed field;
- coordinate-fork loss divided by one challenge-set cardinality;
- the complete base-plus-neighbour fork shape;
- separate weak extractor agreement;
- exact commitment and all-matrix PiDEC checks, with public-input
  recomposition derived from the verifier-computed split;
- Theorem 8 divisor and difference conditions;
- derived expansion `T`, not a typed planning constant;
- commitment width in ring elements, not the CCS row count.

## 4. Reviewed errata result

All 16 errata rows have a locked source and an atomic contract destination.
The review confirmed coverage for norm roots, degree semantics, SumCheck
degree, target shift, CCS zero set, `L_in`, CE types, ambient bound, error
budget, extractor flow and runtime, coordinate-fork loss, PiRLC projection,
PiDEC equations, evaluation notation, and source indexing.

The errata ledger does not classify Nightstream design choices as paper
corrections.

## 5. Appendix B.2 dimension defect

The reviewed paper profile states `d=54` and `n_F=2^30`, but:

```text
2^30 mod 54 = 46.
```

It therefore does not define an integer `n_R=n_F/54`, and it cannot directly
instantiate the common-cube identity construction. The contract records this
as a paper-profile conflict, not accepted errata.

Nightstream resolves the conflict with an artifact-owned ring-aligned full
assignment `z=x||w`. The exact width comes from the verifier key, is at most
16,777,206 fields, and is not inferred from `2^30`.

## 6. Norm-binding decision

The earlier rectangular design did not bind an independent norm terminal to
the witness under the output commitment. A non-square relation matrix cannot
be the paper identity, and a prover-supplied terminal cannot gain authority
from a separate SumCheck alone.

Nightstream v1 selects the lower-risk closure:

```text
logical rows            verifier-key relation artifact
logical z=x||w fields   verifier-key relation artifact, 54-aligned
padded cube             2^24
M_0                     [I_m;0]
application matrices    13
total matrices          14
```

Every logical vector uses a zero-based prefix in the same little-endian Boolean
cube. Application outputs are zero after the logical row count. The lifted
polynomial ignores only the added `M_0` input. Since `P_2(0)=0`, zero padding
does not add a norm violation. The terminal `ct(y_(i,0))` is in the CE output
for the same committed assignment.

This closes the protocol-design ambiguity. Lean proofs for padded identity and
padding refinement remain explicit G2 obligations.

## 7. Removed protocol surface

The selected protocol has no:

- separate FE and NC SumChecks;
- column point or column terminal;
- relation-authoritative `y_carrier`, `s_col`, or `y_zcol`;
- column replay through PiRLC or PiDEC;
- `beta_a`, `beta_r`, or `beta_m` challenge;
- fallback to a legacy Split-NC proof.

The public carrier is the canonical 270-field `x`. A verifier may recompute a
public evaluation as a cache, but that value is not relation authority.

This removal restores the paper challenge and composition surface. It also
prevents three incompatible column-output types from sharing one name.

## 8. Exact v1 arithmetic

The profile checks recompute:

```text
row variables                       24
sources                             1+14 = 15
matrices                            1+13 = 14
polynomial total degree             8
joint individual degree D_Q         9
SumCheck term N_SC                  24*9 = 216
root term D_SZ                      10,599
combined field numerator            10,815
terminal evaluations                15*14 = 210 R_K values
PiRLC coefficients                  15*54 = 810
norm guard                          15*216 = 3,240 < 16,384
Module-SIS infinity bound           8*216*16,384 = 28,311,552
```

Container counts were derived from the same shape:

```text
statement base fields  39,848
proof sections         480, 22,680, 34,776
proof base fields      57,936
```

The old 500-field round section was rejected. The selected joint proof has 24
rounds, 10 extension coefficients per round, and two base fields per extension
coefficient: `24*10*2=480`.

The Ajtai setup profile now fixes the ChaCha8 64-bit counter stream, row-seed
order, fixed chunk formula, chunk-seed order, matrix traversal, 54-word batch,
and replacement order for rejected Goldilocks candidates. Independent Python
checks reproduce the initial, random-access, and multi-chunk Rust/Lean vectors.
The security proof must still state the seeded-PRG assumption.

## 9. Transcript and sampler result

The state machine fixes 12 ordered fold-verifier events, four challenge
families, and five bounded repetitions. Each fold starts a fresh zero-state
duplex. Adjacent folds link by exact typed equality from the ordered PiDEC
children to the next ordered running claims. All folds use one selected
verifier key and profile. The final fold transcript digest
is a verifier-derived receipt, not CE authority or next-fold input. The
recursive transcript schedule fixes every frame payload count, squeeze count,
tag, and loop nesting. The Poseidon2
profile fixes field, width, rate, capacity, rounds, constants, matrices, frame
format, numeric domain tags, direction padding, continuation, extension decoder,
and final fold transcript digest.

The verifier key uses one canonical sparse Structure stream. Its four-field
digest is recomputed by a fresh selected field duplex over the contract domain,
profile version, setup code, dimensions, 32 seed-byte lanes, and the complete
Structure stream. The actual verifier key remains authoritative; its digest is
only a transcript binding.

The circuit public image has nine ordered fields: contract domain, profile
version, container variant, fold index, fold count, and four statement-digest
fields. A fresh selected duplex frames the session, verifier-key digest, and
the 39,848-field canonical statement stream before the statement-digest
squeeze. The checker fixes the exact order, count, tags, and preimage layout.

The PiRLC sampler processes 15 sources, 54 coefficients per source, and up to
three attempts. It accepts `x<q-1` and maps `x mod 5` to the ordered alphabet.
Since `q-1` is divisible by five, the accepted digit is exactly uniform. The
proof rejects after three failed candidates. The per-fold exhaustion bound is
`810/q^3`.

The contract profile is exact enough to implement one deterministic native fold
algorithm. Circuit correspondence is a separate G4 edge, not a later native
verifier event. The Fiat-Shamir and sampler conformance theorems remain G5 and
G3/G4 evidence.

The rejection registry contains only fail-closed checks on supplied or selected
inputs. The norm terminal and PiRLC output are verifier-derived, so the state
machine does not invent proof fields or comparison failures for those values.
For the fixed circuit, the verifier-key digest is a checked circuit constant.
Its manifest and lowering proof must equate that constant to native
recomputation; R1CS does not need to hash the full sparse Structure again.

## 10. Dependency and authority review

The final requirement graph has 104 nodes and 170 direct edges. It is acyclic
and transitively reduced. No rule has mixed paper and Nightstream authority.
Paper identity and the Nightstream padded replacement remain separate authored
facts; the generated contract shows the replacement.

All 20 decisions appear exactly once across the 15 G1 decision claims. Decision
impact is derived from requirement authority. There is no second hand-written
decision-to-rule map. Erratum-to-rule edges are also derived from normative
citations, not copied into the errata table. All five evidence ledgers have
exactly one row for each rule. Unsupported implementation rules remain at
evidence level `none`.

The final terminology audit uses “assignment” for the full committed
`z=x||w` width and “witness” only for `w` or a relation witness. It also
distinguishes the four-field fold transcript receipt from the independent
four-field circuit statement digest.

## 11. Security boundary

The selected policy is at least 96 classical bits, one proof and one session
per verifier key, at most 64 folds, and at most 262,144 adaptive oracle queries.
This limit includes at most 157,313 prescribed
tagged squeezes per key, so the threat model does not exclude an honest
maximum-size proof.
The algebraic planning component is about 114.60 bits for one fold and 108.60
bits after a simple 64-fold union bound. The sampler-abort component is about
182.34 and 176.34 bits for the same cases.

These values exclude open terms for padded-relation extraction, strong-weak
composition, Module-SIS, seeded setup, Poseidon2, Fiat-Shamir, canonical
encoding, adaptive composition, Rust transfer, circuit correspondence, the
terminal proof, and deployed verification. The contract forbids presenting
the planning counts as end-to-end security.

## 12. Package and parser hardening

The 15 reviewed paper files and exact errata patch are immutable package files.
Package-only validation was repeated after the contract directory was copied to
a clean location with no repository `docs/` tree. Source derivation and all
package checks passed there.

The migration command now audits the sealed baseline and completed receipt. It
does not compare the live protocol with the superseded draft. The manifest
excludes only named Python/tool caches, bytecode, and `.DS_Store`; any other
added file still makes the package stale.

The fixed section census gives an exact 318,832-byte statement and
463,528-byte proof. The decoder compares the complete length with checked
arithmetic before payload allocation. The Ajtai rule also states the exact
left matrix-vector commitment equation and rejects transposed or affine forms.

## 13. Residual assurance work

The following items do not block implementation against this specification:

- exact Lean build and proposition review;
- padded-identity and zero-padding theorems;
- strong, weak, extractor, and concrete loss proofs;
- selected joint PiCCS Rust implementation;
- exact transcript and sampler implementation;
- Rust-origin conformance suite and universal refinement;
- selected current-circuit manifest and four correspondence theorems;
- concrete terminal backend manifest and on-chain verifier identity;
- complete lifetime security reduction.

They remain release blockers in G2 through G5.

## 14. Conclusion

No unresolved source-normalization or protocol-design ambiguity remains in the
v1 contract. `PaddedRowIdentity`, its exact profile, and its verifier state
machine form one implementation target. G0B and G1 may close with receipts
bound to the final contract hash, profile hash, and reviewed evidence set.

The present implementation must be changed to match this target. Production
claims remain prohibited until G2 through G5 close.

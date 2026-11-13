# Compression of the final proof over Akita

## Status

Accepted by the owner on 2026-11-13:

- Akita (ePrint 2026/1983) is the polynomial commitment scheme of the
  compression.
- Poseidon2 makes each value that a circuit computes again. Keccak-family
  hashes are used only inside Akita: the Keccak-f duplex sponge, SHA3-256
  digests, the SHAKE128 session identifier and the SHAKE256 setup.
- No recursion over the final proof is planned.

Proposed. Each other entry of this record was selected without the owner and
waits for the owner's review. No Lean statement covers the argument of this
record.

## Problem

The Spartan, WHIR and shrink path was removed
([`field-2p64-59-ring-x64.md`](./field-2p64-59-ring-x64.md)). A proof that
`Verifier::verify` accepts carries the witnesses of both slots and of the
fresh instance. A caller needs a proof without witnesses that a verifier
checks against the same statement `(i, z0, zi)`.

Akita commits and opens a multilinear polynomial. It does not say which
statement to prove, how to prove a norm bound, or how to join its transcript
to a Poseidon2 transcript. This record fixes these choices.

## SuperNeo

SuperNeo v1.2 gives the folding scheme and the CE relation (Definition 21).
Its abstract names no compression of a final proof, and this record uses no
SuperNeo result about compression. The body of the paper was not searched for
this record. The relations that the argument targets are the Lean ones:
`Spec.CE.Holds` for a slot, `Spec.CCS.Holds` for the fresh instance, and the
verifier `Export.BoundedDepth.RunCompleteness.Verifies`.

## Decision

`Prover::compress` makes a final proof from a proof with witnesses, and
`Verifier::verify_final` checks it ([SPEC.md](../SPEC.md) Algorithms 9
and 10). The final proof has the statement, both slot claims with their
depths, the CE claim of the fresh instance, one Π_CCS sum-check, and one
argument over two Akita commitments. It has no witness and cannot be
extended.

| Item | Value |
|---|---|
| Commitment scheme | Akita, git revision `d50f700fb1a292395a717cef0e6f437c7ce2cdb1`, crate version 0.1.0 |
| Akita configuration | `proof_optimized::fp64::Dense`: base field `F_q`, extension `F_q[v]/(v² − 2)` |
| Schedule catalog | `crates/neo-compress/artifacts/fp64_dense.aks`, SHA-256 `44e2a3f5a587a4ea298cea3f74bd7e6568fc8f86d498a3a0c9706fafc0d4cee3` |
| Schedule rows in use | one polynomial of 16, 20, 26, 28, 29 or 30 variables |
| Committed polynomials | `W`: each witness, then the counts of each table. `V`: the lookup inverses |
| Lookup table of slot `s` | `[−accumulatorBound(depth_s), accumulatorBound(depth_s)]`, at most 20,737 values |
| Lookup repetitions `L` | 3 challenges of `F_q`. `V` has room for 4 columns |
| Field map | `a + b·u ↦ a + (b·c)·v` with `c = 6880439088830499533` and `2c² = 7 (mod q)` |
| Transcript domains | `Nightstream/OptionB/compress/ccs/v1` for Π_CCS, `Nightstream/compress/akita/v1` for the argument |
| Rust toolchain | 1.99.1, which Akita requires |
| `jolt-field` | git revision `25d87ba68bde9f4a85ac986f5487b2e7b7db2699`, Akita's own pin |

### What the proof states

1. **No shrink layer.** The verifier checks the argument directly. The removed
   shrink layer made a circuit of a verifier, because the WHIR proofs of the
   layer below it were large. The size of an Akita opening changes little
   with the number of variables, so no such circuit is necessary.
2. **Three CE claims.** The terminal check of SPEC.md Algorithm 6 has two
   slot accumulators and one fresh instance. Each slot accumulator is a CE
   claim (`Lifecycle.BoundedDepth.SlotClaim`). One Π_CCS run with no running
   claim turns the fresh instance into a CE claim at a random point. Its
   sum-check proves each row of the relation and that each coefficient is in
   `{−1, 0, 1}`. This run uses the Π_CCS code of the folds.
3. **Π_CCS binds `W` first.** The Π_CCS transcript absorbs the Akita
   commitment of `W` before its first challenge. So the fresh witness in `W`
   is fixed before the challenges that judge it.
4. **Ring equations at one point.** The Ajtai opening, `Eval_K` and each
   `Eval_A` of a claim are equations `Σ_j P_j · z_j = Y` in
   `K[X]/(X^64 + 1)`. Each side is a reduced polynomial of degree below 64.
   Two different sides agree at a random `δ ∈ K` with probability at most
   `63/|K|`. The left side at `δ` is linear in the coefficients of `z`, with
   the public weights `(X^h · P_j)(δ)`. So each equation is one inner product
   of `W` with public weights, and the proof has no quotient polynomial.
5. **Exact lookup table.** The slot check of Lean is `Spec.normBounded` at
   the bound `accumulatorBound j + 1` for depth `j`
   (`Export.BoundedDepth.StepExtend.Honest`,
   `Spec.GlobalParams.accumulatorBound`). A range of a power of two would
   accept a larger norm than Algorithm 6 accepts. The lookup proves, for a
   challenge `τ`, that `Σ_cells 1/(τ − z) = Σ_values count/(τ − value)`.
6. **Challenges of the base field, with repetitions.** Akita commits a
   polynomial over `F_q`, so the inverses must be in `F_q` and `τ` must be
   in `F_q`. One challenge gives the error `(N + T)/q` for `N` cells and `T`
   table values. `L` independent challenges give `((N + T)/q)^L`.
7. **Counts are cells of `W`.** The counts must be fixed before `τ`. `W` is
   committed before `τ`, so the counts need no third commitment.
8. **Two commitments and two openings.** `V` depends on `τ`, and `τ` depends
   on the commitment of `W`. So two commitments are necessary. Each uses a
   shipped row of the catalog for one polynomial. One sum-check ends in one
   evaluation of `W` and one of `V`.
9. **Transcript.** The Poseidon2 transcript of the Π_CCS run makes each
   challenge of the argument. It absorbs the bytes of each Akita commitment.
   The session bytes of each Akita opening are a label and lanes 0 to 3 of
   the sponge state after the last message. Akita makes its own challenges
   from its Keccak sponge, which no circuit computes.
10. **The verifier is linear in the public data.** It computes the public
    weights at the point of the sum-check with one pass over the Ajtai key,
    which it expands from SHAKE128, and one pass over the rows of each
    relation. `Verifier::verify` reads the same data.
11. **The Akita schedule comes from the verifier.** The verifier must take
    the number of variables of `W` and of `V` from its own layout of the
    claims, and the schedule row from the embedded catalog. A value that the
    proof carries is not authority.
12. **The field map.** `K` and Akita's extension are both the field of `q²`
    elements. The map is a field isomorphism that fixes `F_q`, so an
    evaluation of a polynomial over `F_q` at a point of `K` maps to its
    evaluation at the mapped point.
13. **The fresh zero tail.** The argument commits only the ring blocks of
    each circuit's package, so no other block exists. It does not prove that
    the fresh witness is zero from the width to the end of its last block.
    `Lifecycle.BoundedDepth.BlockSupport.Supported` requires zero only on a
    block that starts at or after the width.
14. **No zero knowledge.** Akita is not zero-knowledge, and the proof
    carries the claims. The final proof hides no witness data in a formal
    sense.

### Error accounting

With `|K| = q²`, `n` variables of `W`, `m = 320` public cells, `t = 4`
matrices and `s = 3` claims, the terms that `neo_compress::error_bits` adds
are these.

| Term | Bound |
|---|---|
| Batch of the equations at `(ω, δ, γ, μ)` | `(κ + 1 + t + max(63, m − 1) + s − 1)/q² = 344/q²` |
| Lookup of each slot | `((N + T)/q)^L` |
| Batch of the inverse checks and the sums | `(2 + n)/q²` |
| Sum-check, degree 3 | `3n/q²` |
| The point for the columns of `V` | `2/q²` |

For the reference application, `N` is `2^23` and `2^25`, `T ≤ 20,737` and
`n ≤ 28`. At `L = 3` the lookups give about `2^−116.97`, the other terms
give `460/q²`, about `2^−119.15`, and the sum is about `2^−116.7`
(computed). `verify_final` and `compress` refuse a layout whose sum is above
the caller's security minimum.

This sum is not a security level of the final proof. It does not include:

- the knowledge error of the two Akita openings. Akita states a target of a
  128-bit Fiat–Shamir rate for each query in the classical random-oracle
  model, and it states no complete bound;
- the error of the Π_CCS run of the fresh instance, which has one source.
  Rust holds the Π_CCS test error for 17 sources to the security minimum
  (SPEC.md section 3);
- the loss of Fiat–Shamir with Poseidon2. Each term above is the error of
  one interactive run;
- Module-SIS for the Akita commitments, which Akita prices at 169 bits of
  quantum Core-SVP cost in the ADPS16 model, and Module-SIS for the Ajtai
  commitments of the claims.

`compress` returns an error when a lookup challenge is a table value. The
probability is at most `L · ΣT/q`, about `2^−47` (computed). The proof is
deterministic, so a new call at the same state fails again. The prover can
extend one step and then compress.

### Consequences

- The verifier is not succinct. Its time is linear in the Ajtai key and in
  the matrices. One pass expands `κ · w · d · 32` bytes of SHAKE128 for `w`
  ring columns, about 18.3 GB for the reference application (computed).
- The project depends on Akita at one git revision. Akita is version 0.1.0,
  has no audit, has no proof in the quantum random-oracle model, and gives
  no compatibility between versions. A change of the revision can change the
  proof bytes.
- The workspace has a second hash family in a proof path, Keccak, inside
  Akita only. The rule of `AGENTS.md` stays in force for the folding scheme
  and the step circuits.
- The Rust toolchain is 1.99.1.
- The prover holds `W`, `V` and the tables of the sum-check in memory. No
  memory limit is stated or tested.
- The final proof is not zero-knowledge.

### Open points

- A succinct verifier. It needs commitments to the Ajtai key and to the
  matrices, and it replaces one function, the public weights at the point
  of the sum-check.
- One batched opening of `W` and `V`. It needs a schedule row that the
  catalog does not ship.
- A Lean statement of the argument and of the Π_CCS run with one source.
- Security against a quantum adversary of the transcript hashes.
- A run at the production cap 64 through the public API. The tests use the
  cap 2.
- A WASM build of the verifier.

## Scope and supersession

This record is the decision record that
[`field-2p64-59-ring-x64.md`](./field-2p64-59-ring-x64.md) names for the
compression layer over Akita. It replaces the removed Spartan, WHIR and
shrink path. It does not change the folds, the step circuits, the packages,
the state hash, or the bytes of a proof with witnesses.

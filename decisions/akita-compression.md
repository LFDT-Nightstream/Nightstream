# Compression of the final proof over Akita

## Status

Accepted by the owner on 2026-11-13:

- Akita (ePrint 2026/1983) is the polynomial commitment scheme of the
  compression. The owner kept it after a review of the architecture.
- Poseidon2 makes each value that a circuit computes again. Keccak-family
  hashes are used only inside Akita: the Keccak-f duplex sponge, SHA3-256
  digests, the SHAKE128 session identifier and the SHAKE256 setup.
- No recursion over the final proof is planned.
- Folding stays. The prover carries only the accumulator between steps.
- The final proof must be a succinct argument of less than 200 KB.
- The final verifier can read the compiled circuit. It must not read
  gigabytes of key material, and it must not read witness openings.
- The committed key. Preprocessing commits the Ajtai key with Akita, and the
  verifier holds the commitment in a verification key.
- The verification key belongs to the generation of the circuit on the
  verifier's side. The verifier never takes it from a proof.

Proposed. Each other entry of this record was selected without the owner and
waits for the owner's review. This includes the form of the key polynomial,
the sum-check that reduces the key's part of the weights, the row point, the
schedule rows, the bytes of the key and their pin, and the four lookup
challenges. No Lean statement covers the argument of this record.

## Problem

The Spartan, WHIR and shrink path was removed
([`field-2p64-59-ring-x64.md`](./field-2p64-59-ring-x64.md)). A proof that
`Verifier::verify` accepts carries the witnesses of both slots and of the
fresh instance. A caller needs a proof without witnesses that a verifier
checks against the same statement `(i, z0, zi)`.

Akita commits and opens a multilinear polynomial. It does not say which
statement to prove, how to prove a norm bound, or how to join its transcript
to a Poseidon2 transcript. This record fixes these choices.

Each CE claim has an Ajtai commitment, so a verifier of the claims needs the
Ajtai key at the point of its check. For the reference application, the key
rows that the circuits read are `18 · 2^19 · 2,048 = 19,327,352,832` bytes of
SHAKE128 output (computed). The owner's requirement is that the verifier
reads none of them.

## SuperNeo

SuperNeo v1.2 gives the folding scheme and the CE relation (Definition 21).
Its abstract names no compression of a final proof, and this record uses no
SuperNeo result about compression. The body of the paper was not searched for
this record. The relations that the argument targets are the Lean ones:
`Spec.CE.Holds` for a slot, `Spec.CCS.Holds` for the fresh instance, and the
verifier `Export.BoundedDepth.RunCompleteness.Verifies`.

## Decision

`Prover::compression_setup` makes the prover's key once,
`Verifier::compression_key` gives the verification key, `Prover::compress`
makes a final proof from a proof with witnesses, and
`Verifier::verify_final` checks it ([SPEC.md](../SPEC.md) Algorithms 9, 10
and 11). The final proof has the statement, both slot claims with their
depths, the CE claim of the fresh instance, one Π_CCS sum-check, and one
argument over Akita. It has no witness and cannot be extended.

| Item | Value |
|---|---|
| Commitment scheme | Akita, git revision `d50f700fb1a292395a717cef0e6f437c7ce2cdb1`, crate version 0.1.0 |
| Akita configuration | `RecursiveCommitmentConfig<proof_optimized::fp64::Dense>`: base field `F_q`, extension `F_q[v]/(v² − 2)`, schedules that offload Akita's own setup |
| Schedule files | `crates/neo-compress/artifacts/key20.aks`, SHA-256 `8ee08715b2120cfb576227f195fde041e5e1dd828511ca5e75d5df73d4c097b5`, and `crates/neo-compress/artifacts/key30.aks`, SHA-256 `b71861d25a4cb23b8eb68b20365bc5b76336224dcacdcada86242c9d9c084f94` |
| Source of the schedule files | Akita's planner at the pinned revision, family `fp64_dense_recursive`, run by `scripts/akita_schedules.sh` |
| Schedule rows | Each row opens three polynomials in one proof. `key20.aks`: a key polynomial of 20 variables, `W` of 16 and `V` of 16, for tests. `key30.aks`: a key polynomial of 30 variables, `W` of 26 or 27 and `V` of 28 |
| Committed polynomials | The key polynomial: the Ajtai key, once, in preprocessing. `W`: each witness, then the counts of each table. `V`: the lookup inverses |
| Key polynomial | Cell `h + 64·j + 64·n·i` holds coefficient `h` of key element `(i, j)`, for `n` blocks, a power of two. Each cell of a row from `κ` is 0 |
| Lookup table of slot `s` | `[−accumulatorBound(depth_s), accumulatorBound(depth_s)]`, at most 20,737 values |
| Lookup repetitions `L` | 4 challenges of `F_q`, one for each column of `V` |
| Field map | `a + b·u ↦ a + (b·c)·v` with `c = 6880439088830499533` and `2c² = 7 (mod q)` |
| Transcript domains | `Nightstream/OptionB/compress/ccs/v1` for Π_CCS, `Nightstream/compress/akita/v2` for the argument |
| Byte tags | `NS-OPTIONB-FIN-2` for a final proof, `NS-COMPRESS-VK-1` for a verification key |
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
   table values. `L` independent challenges give `((N + T)/q)^L`. `V` has
   four columns, and `L = 4` puts the lookup error below each other term.
7. **Counts are cells of `W`.** The counts must be fixed before `τ`. `W` is
   committed before `τ`, so the counts need no third commitment.
8. **Two commitments in a proof.** `V` depends on `τ`, and `τ` depends on
   the commitment of `W`. So the proof carries two commitments. The
   commitment of the key polynomial is in the verification key.
9. **The committed key.** The key's part of the public weights at the point
   of the sum-check is a sum over the key elements that the claims read. The
   proof carries this sum as a claim of the prover. A second sum-check, of
   degree 2 over the lanes and the blocks, reduces the claim to one value of
   the key polynomial, and Akita opens that value against the commitment of
   the verification key. So the verifier reads no key element.
10. **A claim reads a prefix of the blocks.** The number of ring columns of
    a relation is not a power of two. So the weight of the second sum-check
    is the multilinear extension of an equality table on a prefix, which the
    verifier computes in a number of steps linear in the variables. One
    weight batches the three uses of the key, by slot 0, by slot 1 and by
    the fresh instance, with the coefficients that the first sum-check
    gives. So the three uses need no new challenge.
11. **The rows need no round.** The powers of `ω` that batch the `κ` rows
    are a scaled equality table: `ω^i = scale · eq(s, i)` with
    `s_k = ω^(2^k)/(1 + ω^(2^k))` and `scale = Π_k (1 + ω^(2^k))` for
    `k < 5`. So the batch of the rows is `scale` times the key polynomial at
    the row point `s`. The identity is exact. No proof exists when a factor
    of `scale` is 0, that is when `ω^(2^k) = −1`. The 2-power roots of unity
    of `K` form a group of order 8, because `q² − 1` is divisible by 8 and
    not by 16. So 7 values of `ω` have no row point, and the probability is
    `7/q²`.
12. **One key for each circuit size.** The key polynomial holds the blocks
    of the larger relation up to a power of two. It is a function of the
    public seeds and of that power of two only. So each circuit of one size
    has the same verification key, and a prover and a verifier of one
    circuit agree on the key.
13. **One Akita proof.** One proof opens the key polynomial, `W` and `V`.
    Each schedule row fixes the commitment profile of the three polynomials.
    Each row of one file has the same profile for the key polynomial, so one
    commitment of the key serves each row.
14. **Akita's own setup is committed too.** The schedule rows offload
    Akita's setup (ePrint 2026/1983 section 9). The verifier derives a short
    prefix of that setup from Akita's public seed. The verification key
    holds Akita's commitments of the longer prefixes.
15. **Transcript.** The Poseidon2 transcript of the Π_CCS run makes each
    challenge of the argument. After the domain of the argument, it absorbs
    the bytes of the verification key, then the claims and the bytes of each
    Akita commitment of the proof. The session bytes of the Akita opening
    are a label and lanes 0 to 3 of the sponge state after the last message.
    Akita makes its own challenges from its Keccak sponge, which no circuit
    computes.
16. **The verification key is pinned, so it is checked, not trusted.**
    Item 12 gives one key for each block count that has schedule rows. So
    the code pins the bytes of each such key, and a test computes each
    pinned key again from the public seeds. `Verifier::compression_key`
    gives the pinned key of the verifier's circuit size, and
    `Verifier::decode_compression_key` accepts loaded bytes only when they
    equal that key. No proof and no value of the prover carries the key.
    Under another key, the commitment of the key polynomial or Akita's setup
    prefixes are not the public ones, and Akita gives no binding. So the
    decoder does not accept a key whose form alone is correct.
17. **The Akita schedule comes from the verifier.** The verifier must take
    the number of variables of each polynomial from its key and its own
    layout of the claims, and the schedule row from the embedded files. A
    value that the proof carries is not authority.
18. **The field map.** `K` and Akita's extension are both the field of `q²`
    elements. The map is a field isomorphism that fixes `F_q`, so an
    evaluation of a polynomial over `F_q` at a point of `K` maps to its
    evaluation at the mapped point.
19. **The fresh zero tail.** The argument commits only the ring blocks of
    each circuit's package, so no other block exists. It does not prove that
    the fresh witness is zero from the width to the end of its last block.
    `Lifecycle.BoundedDepth.BlockSupport.Supported` requires zero only on a
    block that starts at or after the width.
20. **No zero knowledge.** Akita is not zero-knowledge, and the proof
    carries the claims. The final proof hides no witness data in a formal
    sense.

In the Nova family, an inner-product argument over Pedersen commitments has
a verifier whose time is linear in the commitment key, and Dory keeps a
commitment to the generators in the verifier's key. Items 9 to 16 are the
lattice form of the second design, with a sum-check and an Akita opening in
place of Dory's pairing check.

WHIR with the same method was not selected. The estimate for its final
proof was 300 to 450 KB, which is above the owner's limit. No such proof was
built or measured.

### Error accounting

With `|K| = q²`, `n` variables of `W`, `n_A` block variables of the key
polynomial, `m = 320` public cells, `t = 4` matrices and `s = 3` claims, the
terms that `neo_compress::error_bits` adds are these.

| Term | Bound |
|---|---|
| Batch of the equations at `(ω, δ, γ, μ)` | `(κ + 1 + t + max(63, m − 1) + s − 1)/q² = 344/q²` |
| Lookup of each slot | `((N + T)/q)^L` |
| Batch of the inverse checks and the sums | `(2 + n)/q²` |
| First sum-check, degree 3 | `3n/q²` |
| The point for the columns of `V` | `2/q²` |
| Key sum-check, degree 2 over 6 lane variables and `n_A` block variables | `2 · (6 + n_A)/q²` |

For the reference application, `N` is `2^23` and `2^25`, `T ≤ 20,737`,
`n ≤ 27` and `n_A = 19`. At `L = 4` the lookups give about `2^−155.99`, the
other terms give `(398 + 4n)/q² = 506/q²` at `n = 27`, about `2^−119.02`,
and the sum is about `2^−119.02` (computed). `verify_final` and `compress`
refuse a layout whose sum is above the caller's security minimum.

This sum is not a security level of the final proof. It does not include:

- the knowledge error of the Akita opening. Akita states a target of a
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

The argument has these premises about the verification key. The key is the
key of SPEC.md Algorithm 11. The pin of item 16 and the test that computes
each pinned key again make this true. Akita is binding and knowledge sound for the key
polynomial and for its own setup prefixes, so the value that Akita opens is
the value of the key polynomial. The verifier never reads the polynomial
([security model](../formal/nightstream-fprime/SECURITY_MODEL.md), trust
item 7).

`compress` returns an error when a lookup challenge is a table value or
when `ω` has no row point. The probability of the first event is at most
`L · ΣT/q`, about `2^−46.7`, and of the second `7/q²`, about `2^−125.19`
(computed). The proof is deterministic, so a new call at the same state
fails again. The prover can extend one step and then compress.

### Consequences

- The verifier reads no element of the Ajtai key. Its time is not linear in
  the key.
- The verifier is not succinct. Its time is linear in the matrices of the
  circuits, because it makes one pass over the rows of each claim's
  relation.
- A verifier expands no key row to get its verification key, because the
  key is pinned. The test of the pin expands the key rows once for each
  pinned key.
- A change of the Akita revision, of the schedule files or of the seeds
  changes the verification key, so the pinned bytes must change with it.
- The prover holds the key rows and their commitment for each `compress`.
- The schedule files have rows only for a key polynomial of `2^9` blocks,
  for tests, and of `2^19` blocks with `W` of 26 or 27 variables and `V` of
  28. A circuit of another size needs new rows from
  `scripts/akita_schedules.sh`.
- The tests that compress a step of the reference application hold the
  prover's key and are ignored. CI runs the public test
  `hash_chain_compresses_and_verifies_without_the_ajtai_key` and the test
  of the large pin alone (`scripts/ci_rust_tests.sh`). The other tests at
  the production size run by hand. CI runs the edits of the argument on
  small claims only.
- The project depends on Akita at one git revision. Akita is version 0.1.0,
  has no audit, has no proof in the quantum random-oracle model, and gives
  no compatibility between versions. A change of the revision can change the
  proof bytes, the schedule files and the verification key.
- The workspace has a second hash family in a proof path, Keccak, inside
  Akita only. The rule of `AGENTS.md` stays in force for the folding scheme
  and the step circuits.
- The Rust toolchain is 1.99.1.
- The prover holds `W`, `V`, the key rows and the tables of both sum-checks
  in memory. No memory limit is stated or tested.
- The final proof is not zero-knowledge.

### Open points

- A commitment to the matrices of the circuits. With it, the verifier makes
  no pass over the rows, and it replaces one function, the matrices' part of
  the public weights at the point of the sum-check.
- A review of the soundness of the committed key by a party outside the
  project.
- A Lean statement of the argument, of the key reduction and of the Π_CCS
  run with one source.
- Security against a quantum adversary of the transcript hashes.
- A compression with a slot depth near the cap 64, which has the largest
  lookup table. The tests compress after 5 steps or less.
- A WASM build of the verifier.

## Scope and supersession

This record is the decision record that
[`field-2p64-59-ring-x64.md`](./field-2p64-59-ring-x64.md) names for the
compression layer over Akita. It replaces the removed Spartan, WHIR and
shrink path. It does not change the folds, the step circuits, the packages,
the state hash, or the bytes of a proof with witnesses.

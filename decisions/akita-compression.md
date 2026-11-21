# Compression of the final proof over Akita

## Status

Accepted by the owner on 2026-11-13:

- Akita (ePrint 2026/1983) is the polynomial commitment scheme of the
  compression.
- Poseidon2 makes each value that a circuit computes again. Keccak-family
  hashes are used only inside Akita: the Keccak-f duplex sponge, SHA3-256
  digests, the SHAKE128 session identifier and the SHAKE256 setup.
- No recursion over the final proof is planned.
- Folding stays. The prover carries only the accumulator between steps.
- The final proof must be a succinct argument of less than 200 KB.
- The final verifier can read the compiled circuit. It must not read
  gigabytes of key material, and it must not read witness openings.
- The matrices of the circuits are not committed. The verifier keeps its
  pass over the compiled circuit, because a commitment adds prover time to
  each `compress`.
- No Rust code of the compression is translated to Lean with Aeneas.
- Preprocessing commits the Ajtai key with Akita, and the verifier holds the
  commitment in a verification key.
- The verification key belongs to the generation of the circuit on the
  verifier's side. The verifier never takes it from a proof.

Accepted by the owner on 2026-11-16: a circuit whose proofs never reach a
refresh fold, such as each WASM size, has a final proof of two claims, and
its key holds the light relation only
([`wasm-light-compression.md`](./wasm-light-compression.md)).

Proposed: each other entry of this record. The owner did not select these
entries, and they wait for the owner's review. They include the form of the
key polynomial, the sum-check that reduces the key's part of the weights,
the row point, the schedule rows, the bytes of the key and their pin, and
the four lookup challenges.

## Problem

A proof that `Verifier::verify` accepts carries the witnesses of both slots
and of the fresh instance. A caller needs a proof without witnesses that a
verifier checks against the same statement `(i, z0, zi)`. The earlier
Spartan, WHIR and shrink path is removed
([`field-2p64-59-ring-x64.md`](./field-2p64-59-ring-x64.md)).

Akita commits and opens a multilinear polynomial. It does not say which
statement to prove, how to prove a norm bound, or how to join its transcript
to a Poseidon2 transcript. This record fixes these choices.

Each CE claim has an Ajtai commitment, so a verifier of the claims needs the
Ajtai key at the point of its check. For the reference application, the key
that the circuits read has `18 · 2^19` ring elements of 64 coefficients. The
owner's requirement is that the verifier reads none of them.

## SuperNeo

SuperNeo v1.2 gives the folding scheme and the CE relation (Definition 21).
This record uses no SuperNeo result about compression. The relations that
the argument targets are the Lean ones: `Spec.CE.Holds` for a slot,
`Spec.CCS.Holds` for the fresh instance, and the verifier
`Export.BoundedDepth.RunCompleteness.Verifies`.

## Decision

`Prover::compression_setup` makes the prover's key once,
`Verifier::compression_key` gives the verification key, `Prover::compress`
makes a final proof from a proof with witnesses, and
`Verifier::verify_final` checks it. [SPEC.md](../SPEC.md) Algorithms 9, 10
and 11 are the rule, with the parameters of section 3 and the transcript of
section 5.4. The final proof has the statement, the slot claims of its mode
with their depths, the CE claim of the fresh instance, one Π_CCS sum-check,
and one argument over Akita. It has no witness and cannot be extended. The
mode has both slots, or slot 0 alone for a circuit whose proofs never reach
a refresh fold (`Export.BoundedDepth.FinalVerifier.Mode`).

| Item | Value |
|---|---|
| Commitment scheme | Akita, git revision `d50f700fb1a292395a717cef0e6f437c7ce2cdb1`, crate version 0.1.0 |
| Akita configuration | `RecursiveCommitmentConfig<proof_optimized::fp64::Dense>`: base field `F_q`, extension `F_q[u]/(u² − 2)`, which is `K`, schedules that offload Akita's own setup |
| Schedule files | `crates/ns-compress/artifacts/key20.aks`, SHA-256 `8ee08715b2120cfb576227f195fde041e5e1dd828511ca5e75d5df73d4c097b5`, and `crates/ns-compress/artifacts/key30.aks`, SHA-256 `b71861d25a4cb23b8eb68b20365bc5b76336224dcacdcada86242c9d9c084f94`, from Akita's planner at the pinned revision (`scripts/akita_schedules.sh`) |
| Committed polynomials | The key polynomial: the Ajtai key, once, in preprocessing. `W`: each witness, then the counts of each table. `V`: the lookup inverses |
| Lookup table of slot `s` | `[−accumulatorBound(depth_s), accumulatorBound(depth_s)]`, at most 20,225 values |
| Lookup challenges `L` | 4 challenges of `F_q`, one for each column of `V` |
| Transcript domains | `Nightstream/OptionB/compress/ccs/v1` for Π_CCS, `Nightstream/compress/akita/v2` for the argument |
| Byte tags | `NS-OPTIONB-FIN-2` for a final proof of both slots, `NS-OPTIONB-FIN-L` for a final proof of slot 0 alone, `NS-COMPRESS-VK-1` for a verification key |
| Rust toolchain | 1.99.1, which Akita requires |
| `jolt-field` | git revision `25d87ba68bde9f4a85ac986f5487b2e7b7db2699`, Akita's own pin |

### The choices

1. **No shrink layer.** The verifier checks the argument directly. The size
   of an Akita opening changes little with the number of variables, so no
   circuit of a verifier is necessary.
2. **Three or two CE claims.** The terminal check of Algorithm 6 has two
   slot accumulators and one fresh instance. Each slot accumulator is a CE
   claim (`Lifecycle.BoundedDepth.SlotClaim`). One Π_CCS run with no running
   claim turns the fresh instance into a CE claim at a random point. This run
   uses the Π_CCS code of the folds. In the light mode, slot 1 is the default
   slot at depth 0, which the verifier builds, so the argument has slot 0
   and the fresh instance of the light circuit, and the key holds the light
   relation only.
3. **Π_CCS binds `W` first.** The Π_CCS transcript absorbs the Akita
   commitment of `W` before its first challenge.
4. **Ring equations at one point.** The Ajtai opening, `Eval_K` and each
   `Eval_A` of a claim are equations in `K[X]/(X^64 + 1)`. Two different
   sides agree at a random `δ ∈ K` with a probability of at most `63/|K|`.
   So each equation is one inner product of `W` with public weights, and the
   proof has no quotient polynomial.
5. **Exact lookup table.** The slot check of Lean is `Spec.normBounded` at
   the bound `accumulatorBound j + 1` for depth `j`
   (`Export.BoundedDepth.StepExtend.Honest`,
   `Spec.GlobalParams.accumulatorBound`). A range of a power of two would
   accept a larger norm than Algorithm 6 accepts.
6. **Lookup challenges of the base field, with repetitions.** Akita commits
   a polynomial over `F_q`, so the inverses and the challenge `τ` are in
   `F_q`. One challenge gives the error `(N + T)/q` for `N` cells and `T`
   table values, and `L = 4` independent challenges give `((N + T)/q)^4`,
   which is below each other term.
7. **Counts are cells of `W`.** `W` is committed before `τ`, so the counts
   need no third commitment. `V` depends on `τ`, so a proof carries two
   commitments. The commitment of the key polynomial is in the verification
   key.
8. **The committed key.** The key's part of the public weights is a sum over
   the key elements that the claims read. The proof carries this sum as a
   claim of the prover. A second sum-check, of degree 2 over the lanes and
   the blocks, reduces the claim to one value of the key polynomial, and
   Akita opens that value against the commitment of the verification key.
   So the verifier reads no key element. One weight batches the three uses
   of the key with the coefficients of the first sum-check.
9. **The rows need no round.** The powers of `ω` that batch the `κ` rows are
   a scaled equality table, so the batch of the rows is a scale times the
   key polynomial at a row point. No proof exists for 7 values of `ω`, with
   probability `7/q²`. `compress` then returns an error.
10. **One key for each circuit size.** The key polynomial is a function of
    the public seeds and of the block count as a power of two only. So each
    circuit of one size has the same verification key. The block count is
    that of the larger relation, or of the light relation in the light
    mode.
11. **One Akita proof.** One proof opens the key polynomial, `W` and `V`.
    The schedule rows offload Akita's setup (ePrint 2026/1983, Section 9),
    and the verification key holds Akita's commitments of the setup
    prefixes.
12. **Transcript.** The Poseidon2 transcript of the Π_CCS run makes each
    challenge of the argument. It absorbs the bytes of the verification key,
    the claims and the bytes of each Akita commitment. Akita makes its own
    challenges from its Keccak sponge, which no circuit computes.
13. **The verification key is pinned.** The code pins the bytes of the key
    of each block count that has schedule rows, and a test computes each
    pinned key again from the public seeds. `Verifier::compression_key`
    gives the pinned key, and `Verifier::decode_compression_key` accepts
    loaded bytes only when they equal that key. Under another key Akita
    gives no binding, so the decoder does not accept a key whose form alone
    is correct.
14. **The Akita schedule comes from the verifier.** The verifier takes the
    number of variables of each polynomial from its key and its own layout
    of the claims, and the schedule row from the embedded files. A value
    that the proof carries is not authority.
15. **The fresh zero tail.** The argument commits only the ring blocks of
    each circuit's package. It does not prove that the fresh witness is zero
    from the width to the end of its last block.
    `Lifecycle.BoundedDepth.BlockSupport.Supported` requires zero only on a
    block that starts at or after the width.
16. **No zero knowledge.** Akita is not zero-knowledge, and the proof
    carries the claims.

Two alternatives were not selected. The estimate for a final proof over WHIR
with the same method was 300 to 450 KB, which is above the owner's limit. A
commitment to the matrices of the circuits removes the verifier's pass over
the rows, but each `compress` must then also prove the matrices' part of the
public weights.

### Error and premises

[SPEC.md](../SPEC.md) section 3 gives the error terms of the argument and
the gate that Rust computes from them (`ns_compress::error_bits`). Lean
bounds the argument in a model with ideal Akita openings: a prover with `Q`
queries has an error of at most `Q · error`, with `error < 2^−119` at the
production shapes (`Spec.Compression.Verifier.soundness`,
`Spec.Compression.Verifier.production_error_lt`). This is not a security
level of the final proof. It does not include:

- the knowledge error of the Akita opening for the pinned setup. This bound
  is an open premise of the final claim. Akita states a target of a 128-bit
  Fiat–Shamir rate for each query in the classical random-oracle model;
- the error of the Π_CCS run of the fresh instance; and
- Module-SIS for the Akita commitments, which Akita prices at 169 bits of
  quantum Core-SVP cost in the ADPS16 model, and Module-SIS for the Ajtai
  commitments of the claims (`[MSIS]`).

The argument has these premises about the verification key. The key is the
key of Algorithm 11, which the pin of choice 13 gives. Under `[AKITA]`, the
commitment binds the key polynomial. The extraction bound for this setup,
including its offloaded prefixes, is open. The verifier never reads the
polynomial, and no Lean statement covers the key
([security model](../formal/nightstream-fprime/SECURITY_MODEL.md), adversary
6 and trust item 7).

### Consequences

- The verifier reads no element of the Ajtai key. It is not succinct: its
  time is linear in the matrices of the circuits.
- A change of the Akita revision, of the schedule files or of the seeds
  changes the verification key, so the pinned bytes must change with it.
- The schedule files have rows only for a key polynomial of `2^9` blocks,
  for tests, and of `2^19` blocks. A circuit of another size needs new rows
  from `scripts/akita_schedules.sh`.
- Akita is version 0.1.0, has no audit, has no proof in the quantum
  random-oracle model, and gives no compatibility between versions.
- The workspace has a second hash family in a proof path, Keccak, inside
  Akita only. The rule of `AGENTS.md` stays in force for the folding scheme
  and the step circuits.

### Open points

- A review of the soundness of the committed key by a party outside the
  project.
- A proof that the Rust layout and the Rust verifier are the Lean models.
  Only tests join them
  ([security model](../formal/nightstream-fprime/SECURITY_MODEL.md), trust
  item 7).
- Security against a quantum adversary of the transcript hashes.
- A WASM build of the verifier.

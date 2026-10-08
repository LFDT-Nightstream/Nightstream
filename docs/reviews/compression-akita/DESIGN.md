# Compression step for Nightstream: Akita analysis and design (2026-10-07)

Status: design for owner review. No code exists for it. All costs are estimates
unless a line says "measured". Code references are at `origin/claude/metal-commit-once`
(e402a6d99).

## 1. The answer in short

- **Akita cannot be used as it is.** It supports only power-of-two rings `X^d+1` over
  three fixed primes (2^32-99, 2^64-59, 2^128-2^32+22537). Goldilocks and our `Phi_81`
  ring are not supported. Over Goldilocks, `X^d+1` splits completely for every
  power-of-two `d`, so Akita's challenge rule gives nothing there.
- **Akita cannot open our commitments.** It opens only its own two-level commitments
  ("cannot resize or reinterpret an incoming commitment", paper p56). Its commitments
  are not linearly homomorphic, so they cannot replace ours inside folding.
- **Its hashes break our rule.** Its transcript is Blake2b or Keccak, and its setup and
  challenges use SHAKE256. Our rule is Poseidon2 only.
- **Two Akita ideas are useful for any compressor:**
  1. *Setup offloading*: commit a large public matrix once, at setup, and prove against
     that commitment. For us, this removes the 967M-coefficient Ajtai key and the CCS
     matrices from the verifier.
  2. *Quotient-lift ring check*: check a ring identity as a polynomial identity with a
     quotient, at a random point. For our key this needs only 22 quotients in total.
- **Recommendation.** Build the compressor that the SuperNeo paper suggests: one last
  native fold, then a Spartan-style argument over Goldilocks with a hash-based
  commitment (WHIR) and Poseidon2 only. Use the two Akita ideas above. An Akita-backed
  compressor is possible (smaller proof) but needs four policy exceptions and an
  unaudited dependency fork. A lattice-native opening of our own commitments is new
  research.

## 2. Akita facts (checked)

| Item | Fact | Source |
| --- | --- | --- |
| Paper | Dao, Bodaghi, Khajehpour, Vitto, Badakhshan, Georghiades, Liu, Zhang, Thaler; ePrint 2026/1983, revised 2026-09-18 | eprint |
| What it is | Lattice multilinear polynomial commitment from Module-SIS | abstract |
| Ring | `Z_q[X]/(X^d+1)`, `d` a power of two only | paper p27 |
| Primes | 2^32-99 (deg-4 ext), 2^64-59 (deg-2 ext), 2^128-2^32+22537 | p173, p175, code |
| Commitment | Two-level Ajtai (inner `t = A s`, outer `u = B G^-1(t)`), then two small binary SIS maps to 128 bytes; not homomorphic | p50-54, p158 |
| Foreign commitments | Not supported | p56 |
| Opening | Recursive Hachi-style folds, digit range check, fused sum-check, quotient-lift ring check, batching | p25, p56-65 |
| Verifier | O(N^{1/K}) with setup offloading (Thm 9.3) | p87 |
| p64, 2^29 coefficients | commit 3.77 s, open 1.78-2.03 s (8 threads), verify 14.7-21.1 ms, proof 68.5-71.8 KB | p129, p177-179 |
| Hashes | Blake2b512 or Keccak transcript, SHAKE256 setup and challenges, proof-of-work grinding | code |
| Security | Classical ROM only (no QROM proof), no zero knowledge, not audited | p111, p132, SECURITY.md |
| Code | LayerZero-Labs/akita v0.1.0 (2026-09-09), Apache-2.0 OR MIT, ~248k lines, no compatibility promise | repo |
| Lattice Jolt | Whole PIOP over fp128; Akita opens all objects in one batch; proofs under 100 KB; 1.3-2.2x faster proving and 2.2-7.4x faster verifying than Jolt-with-Dory. On-chain verification exists only as draft PRs (BN254 + HyperKZG) | paper 13.4, jolt repo |
| With folding | No public work combines Akita or Hachi with Neo, SuperNeo, LatticeFold or Symphony | search |

Our ring over Goldilocks: `p mod 81 = 4`, `ord_81(p) = 27`, so `Phi_81` splits into
two irreducible factors of degree 27 (computed). This is why our own strong challenge
set works and Akita's does not transfer.

## 3. What a compressor must prove

Today the terminal verifier (`crates/nightstream/src/lifecycle/verify.rs:47-247`)
recomputes all 17 Ajtai commitments from the witnesses and streams every matrix row.
The shipped terminal proof is about 221.7 MB. A compressor must prove the same
statement without the witnesses:

- 16 running claims and 1 fresh claim: each witness is small, its Ajtai commitment is
  the claimed `C`, its public projection is the claimed `X` or `x`;
- the 16 running claims open to their `Eval_K` and `Eval_A` at the shared point `r`;
- the fresh witness satisfies the CCS `a*b - c + s^7 = 0` on all 1,004,131 rows.

The state hash, `encHash` and the canonical-children check need no witness. They stay
as native verifier checks.

**All three design candidates agree on the core:**

1. **One last native fold, without `Pi_DEC`.** `Pi_CCS` then `Pi_RLC` turns the 17
   claims into one CE(B) claim (SuperNeo Theorem 12, already used by every fold). The
   honest parent witness has `|z| <= 17 * 216 = 3,672 < B = 2^16`. The verifier
   checks the sum-check and combines the commitments homomorphically. This removes the
   CCS (and its degree 8) from the argument, and it cuts the committed data 17x.
2. **Commit the Ajtai key and the CCS matrices once, at setup** (Akita's offloading
   idea, SPARK-style for the matrices). The verifier never expands SHAKE and never
   streams matrix rows.
3. **Check `C = A z` and the evaluations as ring-linear identities with quotients** at a
   random point. All 27 rows (22 Ajtai rows, 5 evaluation rows) batch into one inner
   product.

They differ in the commitment scheme that commits `z` and the setup data.

## 4. The three candidates

| | 1. Native hash-based (WHIR) | 2. Akita-backed | 3. Lattice-native opening |
| --- | --- | --- | --- |
| Field | Goldilocks, `K = F_{p^2}`, plus `E = F_{p^4}` for range and WHIR | fp128, Goldilocks values as small integers; checks `Phi_81` at its 54 roots in fp128 (`q = 1 mod 81`) | Goldilocks, our ring `Phi_81` |
| Commitment | WHIR, Poseidon2 Merkle trees | Akita, forked transcript | New two-level Ajtai over `Phi_81`, recursive folds |
| Proof size | ~1.6 MB (less with grinding) | ~0.44 MB | ~0.3-0.6 MB |
| Verifier | ~0.1 s, ~41k Poseidon2 permutations | ~30-55 ms | ~0.06-0.3 s |
| Extra prover time | ~15-20 s Metal, ~75 s CPU | ~20-35 s CPU, 25-40 GB RSS | ~25-50 s Metal |
| Setup | 3-4 min, ~63 GB prover data on disk | ~6 s, Akita handles | minutes |
| Poseidon2-only | Yes | Only after a fork; SHAKE in setup | Yes |
| Non-native arithmetic | None | Yes (bounded, with quotients) | None |
| Paper-faithful | Paper's own suggested route ("Spartan + FRI-based PCS"); the argument is a new component | Contradicts the paper's stated route | Steps after the fold are new protocol |
| Main risk | Merged multi-size WHIR tail; Poseidon2 Merkle throughput | Unaudited v0.1.0 fork; owner exceptions | No published proof for these folds over a non-power-of-two ring |

## 5. Synthesis decision

**Base: candidate 1 (native hash-based).** It is the only shape that satisfies every
hard rule without an exception: Poseidon2 only, native Goldilocks, no new lattice
assumption, no unaudited dependency, and the route the paper names. Its verifier uses
only Goldilocks arithmetic and Poseidon2, so a later on-chain wrapper does not need
non-native arithmetic.

Taken from candidate 2: the observation that the fold must happen before the argument
(the CCS never enters it), and that the carried accumulator statement (~233-262 KB)
stays in clear because the state hash needs it. Rejected: the fp128 field and the
Akita fork (policy exceptions, dependency risk). Kept as the fallback if proof size
must drop below ~0.5 MB and the owner grants the exceptions.

Taken from candidate 3: a staged plan. Milestone 1 ships a small proof with a verifier
that evaluates the key and the matrices itself (no setup data, no SPARK). Milestone 2
adds the setup commitments. Rejected: the new lattice opening (research risk, new MSIS
instances, not paper-faithful).

## 6. Recommended design

### Usage (caller's view)

```rust
use nightstream::{Circuit, CompressedVerifier, CompressionData, Engine, VerifyingKey};

// Once per circuit: deterministic setup. Writes the prover data, returns a small key.
let circuit = Circuit::load("app.nsc")?;
let data = circuit.preprocess_compression("app.compress")?;
std::fs::write("app.vk", data.verifying_key().to_bytes())?; // a few hundred bytes

// Prover: build the chain as today, then compress. `proof` stays usable for `extend`.
let prover = circuit.prover(Engine::Metal, minimum_security_bits)?;
let mut proof = prover.prove(z0, &inputs[0])?;
for step in &inputs[1..] {
    proof = prover.extend(&proof, step)?;
}
let small = prover.finish_with_spartan(&proof, &data)?;

// Verifier: no package, no witnesses, no SHAKE.
let key = VerifyingKey::from_bytes(&vk_bytes)?;
let verifier = CompressedVerifier::new(key, minimum_security_bits)?;
let proof = verifier.decode_proof(&small.to_bytes())?;
verifier.verify(&expected_state, &proof)?;
```

### Type sketch

```rust
// crates/nightstream/src/circuit.rs (additions)
impl Circuit {
    /// One-time deterministic setup: commits the Ajtai key prefix and the CCS
    /// matrices. Refuses an existing directory.
    pub fn preprocess_compression(&self, dir: impl AsRef<Path>) -> Result<CompressionData, Error> { unimplemented!() }
}
impl Prover {
    /// Native Pi_CCS + Pi_RLC on the final accumulator, then the CE(B) argument.
    /// Deterministic. The input proof does not change.
    pub fn finish_with_spartan(&self, proof: &Proof, data: &CompressionData) -> Result<CompressedProof, Error> { unimplemented!() }
}

// crates/nightstream/src/compress.rs
//! Owns the public compression types, the native last fold, and the compressed
//! verifier flow. Does not own the argument (neo-spartan) or the terminal statement
//! checks (shared with `Verifier::verify`).
pub struct CompressionData { /* private: neo_spartan::Preprocessed, VerifyingKey */ }
pub struct VerifyingKey { /* private: package identity (a claim), verifier context, shape, Merkle roots, schedule */ }
pub struct CompressedProof { /* private: state, 16 running claims, fresh C and x, fold messages, argument */ }
pub struct CompressedVerifier { /* private: VerifyingKey */ }
impl CompressedVerifier {
    pub fn new(key: VerifyingKey, minimum_security_bits: u32) -> Result<Self, Error> { unimplemented!() }
    pub fn decode_proof(&self, bytes: &[u8]) -> Result<CompressedProof, Error> { unimplemented!() }
    pub fn verify(&self, expected: &State, proof: &CompressedProof) -> Result<(), Error> { unimplemented!() }
}

// crates/neo-spartan (new crate; replaces the deleted wip-spartan, no forked code)
//! Knowledge argument for one SuperNeo CE(B) claim: |z| < B, C = A z over
//! F[X]/Phi_81, x = L_in(z), and the Eval_K / Eval_A openings at r'.
//! Poseidon2 only. The verifier never expands SHAKE.
pub fn preprocess(shape: Shape, matrices: &SparseMatrices, dir: &Path) -> Result<Preprocessed, Error> { unimplemented!() }
pub fn prove(data: &Preprocessed, claim: &ParentClaim, z: &[F], t: &mut Poseidon2Transcript) -> Result<Argument, Error> { unimplemented!() }
pub fn verify(key: &VerifyingKey, claim: &ParentClaim, argument: &Argument, t: &mut Poseidon2Transcript) -> Result<(), Error> { unimplemented!() }
```

Internal modules of `neo-spartan` (each under 600 lines): `layout` (all index math),
`ring_check` (quotient-lift for 27 ring rows), `range` (logup, table `-(B-1)..(B-1)`),
`spark` (matrix evaluation), `gkr` (fractional sums over `E`), `whir` (Poseidon2
Merkle, multi-oracle), `preprocess`, `codec`. `neo-math` gains `E = F_{p^4}`
(`x^4 - 7`, irreducible: 7 is a non-residue and `p = 1 mod 4`).

### Protocol steps

0. **Statement checks (native, no witness):** shapes, shared `r`, canonical children,
   fresh `x = encHash(state hash)`. Reuses the code in `verify.rs`.
1. **Transcript:** one Poseidon2 transcript with a new domain chunk
   (`Nightstream/SuperNeo/compress/v1`) that absorbs the key digest, the state and the
   accumulator.
2. **Last fold:** `Pi_CCS` (existing engine) to 17 claims at one point `r'`, then
   `Pi_RLC` (existing sampler) to one CE(B) parent claim. No `Pi_DEC`.
3. **Commit `z`** (WHIR) and send the 22 + 5 quotient polynomials (degree <= 52).
4. **Batch:** random `alpha` in `K` and row weights turn all 27 ring rows into one
   inner product `sum z(s, l) * W(s, l) = V`.
5. **Inner sum-check** (26 rounds, degree 2, over `K`): leaves three evaluation
   claims: on `z`, on the committed key, on the committed matrices.
6. **Range:** logup over `E` proves `|z| < B = 2^16` (the exact CE(B) norm conjunct,
   which is what makes `C = A z` binding under the existing MSIS assumption).
7. **SPARK:** read-only memory checks for the matrix evaluation.
8. **GKR** over `E` for the fractional sums of steps 6 and 7.
9. **WHIR** over `E`: one proof for all claims on `z`, the SPARK tables, the matrices
   and the key.

Soundness map: steps 0-2 use the existing assumptions (Poseidon2 in the ROM, SuperNeo
Theorem 12, MSIS at the existing norm). Steps 3-5 use Schwartz-Zippel (errors about
`2^-121` to `2^-122`). Steps 6-8 run over `E` because over `K` the logup error would be
only about `2^-100`. Step 9 uses WHIR's published analysis (Johnson regime, queries
set to the target `lambda`).

### Costs (estimates, not measured)

| Item | Estimate | Main driver |
| --- | --- | --- |
| Proof | ~1.6 MB (~262 KB accumulator, ~76 KB fold, ~1.17 MB WHIR) | WHIR queries; grinding or the conjectured regime cut this |
| Verifier | ~0.1 s, ~41k Poseidon2 permutations | WHIR paths |
| Prover | ~15-20 s Metal, ~75 s CPU | Poseidon2 Merkle hashing of the SPARK tables |
| Setup | 3-4 min, ~63 GB on disk | key codeword (34 GB), matrix codeword (17 GB) |
| Prover RAM | ~15 GB | SPARK codeword |

Milestone 1 (verifier evaluates the key and matrices itself) has no setup data and no
SPARK; its verifier costs about 1G field operations (about 1 s), and the proof is
smaller.

## 7. Tradeoffs accepted

- We accept a ~1.6 MB proof in exchange for a Goldilocks-native, Poseidon2-only
  verifier (versus ~0.44 MB with Akita and four policy exceptions).
- We accept ~63 GB of prover setup data in exchange for a verifier that never touches
  the key or the matrices.
- We accept a second challenge field `E = F_{p^4}` in exchange for provable bounds
  without extra queries.
- We accept that the 16 running claims travel in clear (~262 KB), because the state
  hash needs them.

## 8. Alternatives considered

- **No last fold** (prove 16 + 1 claims directly): 17x the committed data, and the
  degree-8 CCS enters the argument. Rejected.
- **Revive `wip-spartan` or the legacy R1CS compilation:** it compiled the Ajtai
  openings into R1CS rows, used a second field crate and a second Poseidon2 instance,
  and a cubic extension that does not contain `K`. Replace it.
- **Akita-backed** (candidate 2): best proof size; rejected for the hash fork, SHAKE
  setup, non-native arithmetic against the paper's stated route, and an unaudited
  v0.1.0 dependency. Fallback.
- **Lattice-native opening** (candidate 3): no second commitment scheme; rejected for
  research risk (no published proof over a non-power-of-two ring) and new MSIS
  instances.

## 9. Decisions for the owner

1. Do you approve a compression argument (quotient-lift, logup range, SPARK, WHIR) as a
   new component? The folding stays exactly the paper's.
2. Is `E = F_{p^4}` acceptable as a compressor-only field?
3. Which WHIR security regime: provable (Johnson bound, no grinding) or with
   proof-of-work grinding or the conjectured regime (smaller proofs)?
4. Is ~63 GB of prover setup data per circuit acceptable, or do we start with
   Milestone 1 (no setup data, ~1 s verifier)?
5. Which on-chain target? A Goldilocks-native chain, or an EVM chain through a wrapper
   proof (a separate project)?
6. If you prefer Akita for proof size: do you approve the Poseidon2 fork of Akita's
   transcript, SHAKE in its setup, its internal digit bases (radix 8 to 64), and a
   non-native decider?
7. CLAUDE.md names `finish_with_spartan`. Keep that name? (Changing CLAUDE.md needs your
   approval.)

## 10. Next step

Measure first, because two numbers decide viability: Poseidon2 Merkle throughput (CPU
and Metal) and the union nonzero count of the four CCS matrices. Then build
`neo-spartan::ring_check` and the inner-product sum-check on a small shape, with tests
that fail when one coefficient of `z` or of a quotient changes.

## 11. Owner decisions (2026-10-07) and consequences

Decisions: the final proof must be post-quantum; about 114 bits is the security
target; a recursion ("shrink") layer is acceptable. Target size: under 100 KB.

Consequences (checked or estimated as marked):

- **Proof size is set by security and the number of openings, not by the key.** WHIR
  paper table (Goldilocks, 128 bits, Johnson bound, rate 1/16): 104 KiB at 2^18,
  128 KiB at 2^22, 150 KiB at 2^26, 163 KiB at 2^28. Under 100 KB needs one final small
  opening, so the shrink layer is required.
- **Setup commitments become required, not optional.** The shrink circuit runs the
  layer-1 verifier. That verifier cannot expand 967M SHAKE key coefficients or walk the
  84.8M matrix nonzeros (57,823,067 + 15,665,400 + 11,356,963 at M5) inside a circuit.
  Milestone 1 stays only as a test step.
- **Setup data is regenerated, not stored (correction of the ~63 GB estimate).** The
  63 GB assumed a padded key layout, a low code rate (for a small layer-1 proof) and
  storage on disk. With the shrink layer, layer-1 size does not matter, so use rate 1/2.
  Commit to the key as 22 row polynomials (43,963,776 coefficients each, padded to 2^26;
  codeword 1 GiB per row) in one Merkle tree whose leaf holds all 22 rows. The prover
  rebuilds the codewords row by row at compression time (SHAKE expansion, NTT, sponge
  states per leaf) in two passes: before the queries and to answer them. Estimate: no
  disk, a few GB of RAM, extra prover time once per chain (not measured; depends on
  Poseidon2 Merkle speed on Metal). The SPARK tables are rebuilt from the package
  in the same way. The verifying key keeps only the roots.
- **`E = F_{p^4}` is required.** Proven Johnson-bound proximity errors at our sizes over
  `K = F_{p^2}` are far below 114 bits (estimate 2^-50 to 2^-80). `p3-goldilocks` has
  extensions of degree 2 (`W = 7`, equal to our `K`) and 5 only; degree 5 does not
  contain `K`. Smallest route: a degree-4 extension (`x^4 - 7`) added to Plonky3, pinned
  by git rev until released.
- **Groth16 and other pairing wrappers are out** (not post-quantum).
- **Capacity-bound WHIR settings are out** (conjecture disproved in its strong form,
  SoK 2026/1367 §5.4). Use the Johnson bound plus proof-of-work grinding.

Layers:

| Layer | Proves | System |
| --- | --- | --- |
| 0 | 16 + 1 claims to one CE(B) claim (Pi_CCS + Pi_RLC, no Pi_DEC) | existing folding |
| 1 | norm, `C = A z`, matrix evaluations | sum-checks + `p3-whir` over `E`, Poseidon2, setup commitments |
| 2 | the layer-1 verifier as a circuit | Spartan + `p3-whir`, rate 1/16, grinding |

Shrink circuit estimate: ~41k Poseidon2 permutations x 150 rows = ~6.2M rows, plus
small `E` arithmetic. Final proof estimate: near 100 KB at 114 bits; a second shrink
layer if the first misses. None of this is measured.

Target chains (owner, 2026-10-07): EVM or Midnight.

- **EVM.** Per-transaction gas cap 2^24 = 16,777,216 (EIP-7825, Fusaka, live
  2025-12-03). Poseidon2-Goldilocks-16 in Solidity: 52,271 gas per permutation
  (measured, trustless-ai/zkIE PR #15); one WHIR opening at 2^22 needs ~1,911
  permutations, ~99.9M gas, so a Poseidon2 final layer cannot fit. Keccak WHIR: ~3.6M
  gas at 2^22 and 100 bits (sol-whir-p3), 3-5M estimated by zkIE. Estimate with ~100 KB
  calldata at 114 bits: 5-8M gas. Keccak is hash-based, so the result stays
  post-quantum. Needs owner approval: Keccak in the layer-2 Merkle tree and transcript
  only.
- **Midnight.** The ledger verifies only its own PLONK/KZG proofs on BLS12-381; no
  precompile or primitive for other proofs (midnightntwrk/servicedesk #203). Option A:
  a Midnight circuit runs our layer-2 verifier (non-native Goldilocks and Poseidon2,
  millions of rows, rough); the chain then checks a KZG proof, so the result is not
  post-quantum. Option B: Midnight adds a native verifier (their protocol decision).
  For option A, the layer-2 hash should stay Poseidon2 (Keccak is far more costly in a
  circuit). Layer 2 takes the hash as its only per-chain choice.

- **Own chain (owner scenario, 2026-10-07): native verifier, up to 150 KB per proof.**
  No gas cap, so Poseidon2 stays everywhere. Layer 2 is still required: the last-fold
  statement alone is over 150 KB (17 commitments x 22 x 54 x 8 B = 161,568 B, plus the
  claims; ~230-262 KB). Layer 2 moves it into the witness; the chain sees the state
  digest and the application's public values. One shrink layer fits: shrink circuit
  ~7M rows (~6.2M for the layer-1 verifier, ~1M for the last-fold check, as in F′),
  witness ~2^23; WHIR (proven, 128 bits, rate 1/16) gives 128 KiB at 2^22 and 140 KiB
  at 2^24, so ~120-135 KB at 114 bits (estimate).

Build order: (1) layer 0; (2) layer 1 with the verifier reading the key (test step);
(3) setup commitments and SPARK; (4) layer 2; (5) on-chain verifier when a chain is
chosen.

## Sources

- Akita paper: https://eprint.iacr.org/2026/1983 (mirror https://eprint.iacr.asia/2026/1983.pdf)
- Akita code: https://github.com/LayerZero-Labs/akita
- Jolt: https://github.com/a16z/jolt (crates/jolt-akita, crates/jolt-spartan-prover, book/src/how/akita.md)
- a16z post: https://a16zcrypto.com/posts/article/lattice-snarks-jolt-post-quantum-faster/
- SuperNeo paper: docs/superneo-paper-v1_2/01_introduction.md:66,179,183
- Hachi notes: docs/Hachi.pdf.md (main checkout)

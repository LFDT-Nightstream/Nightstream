# Glossary

The authoritative paper-symbol → code mapping lives in
`crates/neo-fold-clean/src/paper/mod.rs` and is kept current with the code. This page
is the prose companion: what each term *means* and where it lives.

## Relations and claims

| Term | Meaning | Code |
|---|---|---|
| CCS | Customizable Constraint System — the relation format being folded. A structure `s = ({M_j}, f)` of matrices plus a multivariate polynomial (SuperNeo Def. 11). | `neo_ccs::CcsStructure`, `neo-fold-clean` `paper::relations::Structure` |
| CCS(b, ℒ) | The committed CCS relation: a satisfying low-norm witness plus an Ajtai commitment (Def. 12). | `paper::relations::CcsRelation`, instance = `CcsInstance` |
| MCS / CCS claim | One committed CCS (claim, witness) pair entering a fold. | `neo_ccs::CcsClaim`, `paper::relations::CcsInstance` |
| CE(b, ℒ) / ME claim | Committed-evaluation claim `(c, x, r, {y_j})` — the universal foldable claim shape: "matrix `M_j` times committed witness, evaluated at point `r`, equals `y_j`" (Def. 13). | `neo_ccs::CeClaim`, `paper::relations::CeClaim` |
| Accumulator | The running set of `k` low-norm CE claims (+ witnesses on the prover side) carried between IVC steps. | `construction2::RunningInstance` (U_i, W_i) |

## Reductions (SuperNeo §7)

| Term | Meaning | Code |
|---|---|---|
| Π_CCS | Sum-check reduction: `K` fresh CCS instances + `k` carried CE claims → `K+k` CE claims at one common evaluation point. | `neo_reductions::api`, paper seam in `paper/reductions/pi_ccs.rs` |
| Π_RLC | Random linear combination over the strong sampling set 𝒞 — aggregates the `K+k` claims into **one** CE claim of norm `B = b^k`. | `paper/reductions/pi_rlc.rs` |
| Π_DEC | b-ary decomposition — splits the norm-B claim back into `k` children of norm `b`, keeping the accumulator low-norm. | `paper/reductions/pi_dec.rs` |
| NIFS | The non-interactive folding scheme Construction 2 consumes: `NIFS.P/V = Π_DEC ∘ Π_RLC ∘ Π_CCS`. | `paper/nifs/` |
| Strong sampling set 𝒞 | Challenge set of low-expansion ring elements (Def. 17); rotation form `RotRho`. Expansion factor `T` bounds norm growth (Thm. 9). | `paper/sampling.rs` |
| Folding engine | `Optimized` (production), `PaperExact` (O(2^ℓ) brute-force reference, cross-check only), `OptimizedWithCrosscheck`. | `neo_reductions::api::FoldingMode` |

## IVC (HyperNova §6.3, Construction 2)

| Term | Meaning | Code |
|---|---|---|
| Construction 2 | HyperNova's compiler from a folding scheme to IVC. Specialized here to ℓ = 1 (one step function, `pc = TRIVIAL_PC`). | `paper/construction2/` |
| F′ | The augmented step function: runs the application step *and* re-runs `NIFS.V` on the previous step's instance, then hash-chains the public state. | `paper/f_prime/` (relation + R1CS), `frontends/f_prime/` (encoded image shell) |
| U_i / W_i | Running accumulator instance / witness. | `construction2::RunningInstance` |
| u_i / w_i | Latest (not-yet-folded) instance(s) / witness(es) — the encoding of step i−1. | `construction2::LatestInstance` |
| ProofState | `Initial` (base case, U = u_⊥) or `Active { running, latest }` — structurally tagged so base and recursive cases cannot be confused. | `construction2::ProofState` |
| x_out | The public IVC output: a Poseidon2 hash chain binding `(vk_fs, counters, z_0, z_i, accumulator digest, …)` across steps. | `paper::digest::state_x_out_digest` |
| enc_inst(h) | Bit-decomposition of x_out used in the public boundary encoding. | `construction2::EncInst` |
| vk_fs | Verifier key digest derived from `(params, structure)` at preprocess time. | `construction2::VerifierKey` |
| Chunk | One contiguous run of folds verified by a single terminal fold; multi-chunk histories currently need the audit/decider path. | `State::chunk_count` |

## Commitment and algebra

| Term | Meaning | Code |
|---|---|---|
| Ajtai commitment | Lattice (module-SIS) commitment `c = cf(M·cf⁻¹(Z))`; S-homomorphic, (d,m,B)-binding, pay-per-bit. | `neo-ajtai` |
| R_q | Cyclotomic ring `F_q[X]/Φ_81(X)` with `Φ_81 = X^54 + X^27 + 1`, so `d = φ(81) = 54`. | `neo_math::Rq` |
| cf / cf⁻¹ / ct | Coefficient map ring↔vector, and the constant-term functional. | `neo_math::{cf, cf_inv, ct}` |
| bar(·) | SuperNeo §5 lifted transform enabling `Mz = ct(bar(M)z)` — products over F_q computed via ring arithmetic. | `neo_math::superneo_bar_*` |
| S-action | Action of the ring (as `d×d` rotation matrices) on committed vectors; what makes Π_RLC's challenge-mixing commitment-homomorphic. | `neo_math::SAction`, `neo_ajtai::AjtaiSModule` |
| split_b | Balanced b-ary decomposition (Def. 3); used by Π_DEC and the pay-per-bit embedding. | `neo_math::balanced`, `neo_ajtai::split_b` |
| ‖·‖_∞ | Centered infinity norm; "low-norm" means every entry in `(−b, b)` balanced representation. | `neo_math::balanced` |
| K (field) | Extension field `F_{q²}` — sum-check challenges live here for soundness. | `neo_math::K` |

## Verification and compression

| Term | Meaning | Code |
|---|---|---|
| Terminal-only verification | `verify_uncompressed`: re-runs only the terminal fold; accepts single-chunk chains. | `lifecycle/verify.rs` |
| Audit verification | `verify_uncompressed_audit`: linear-time replay of every step; catches audit-trail tampers; required for multi-chunk. | `lifecycle/verify.rs` |
| Decider | The terminal check of the folded accumulator. Statement contract in `paper/decider.rs`; full-history audit R1CS in `engine/decider.rs`; compact Spartan proof pending (PR5). | see [Decider](architecture/decider.md) |
| Terminal CE | The relation the decider must establish on the final accumulator claims: commitment opening, public-input projection, low norm, `y_ring = (M·z)(r)`, `ct = lane0(y_ring)`. | `paper/decider_ce_relation/`, `paper/terminal_ce/` |
| PublicImage | The chain-binding public coordinates a verifier recomputes (vk_fs digest, counters, z_0, z_i, pc, acc_digest, public trace, x_out). | `paper/decider.rs` |
| Spartan2 | Vendored sum-check-based SNARK used as the terminal compression backend. | `crates/spartan2` |

# SuperNeo Folding (Π_CCS → Π_RLC → Π_DEC)

SuperNeo §7 defines Neo's folding scheme for CCS in the lattice setting. One fold
takes `K` fresh committed CCS instances plus the carried accumulator of `k` CE claims
and outputs a new accumulator of `k` low-norm CE claims. The composition of the three
reductions is what `neo-fold-clean` calls `NIFS` (`src/paper/nifs/`).

## The relations (SuperNeo §7.1)

| Relation | Content | Code |
|---|---|---|
| Structure (Def. 11) | `s = ({M_j}_{j∈[t]}, f)` — `t` matrices and a degree-`u` polynomial `f`. CCS is satisfied when `f(M_1·z, …, M_t·z) = 0` row-wise. | `paper::relations::Structure`, `neo_ccs::CcsStructure` |
| CCS(b, ℒ) (Def. 12) | A committed instance: Ajtai commitment `c` to the low-norm witness matrix `Z`, public input `x`, satisfying the structure. | `paper::relations::CcsInstance` |
| CE(b, ℒ) (Def. 13) | Committed evaluation: `(c, x, r, {y_j})` claims that for the committed `Z`, the multilinear extension of `M_j·z` evaluated at point `r` equals `y_j` — with `‖Z‖_∞ < b`. | `paper::relations::CeClaim` |

CE is the *universal foldable claim*: both reduction outputs and the accumulator have
this shape, which is what makes the scheme compose indefinitely.

## Π_CCS — the sum-check fold (§7.3)

Reduces `CCS^K + CE(b)^k` to `K+k` CE claims sharing one evaluation point `r′`.
The prover runs a sum-check over a batched polynomial `Q` (built from the CCS
constraint polynomial and the carried evaluation claims, mixed by challenges α, γ);
the verifier checks the sum-check transcript and the terminal identity, then emits the
claims `y′_{i,j}` at `r′`.

- The math lives in `neo-reductions` (`api::pi_ccs_prove` / `pi_ccs_verify`, optimized
  and paper-exact engines).
- The paper layer in `neo-fold-clean` exposes only a shape-checked seam:
  `paper/reductions/pi_ccs.rs` wrapping `engine/optimized.rs`.
- Sum-check soundness comes from the extension field `K = F_{q²}`; per-shape effective
  λ is validated at preprocessing (see [Parameters](parameters.md)).

## Π_RLC — random linear combination (§7.4)

Aggregates the `K+k` CE claims into **one** CE claim by mixing with challenges
`ρ_i` drawn from the strong sampling set 𝒞 (Def. 17) — in code, rotation matrices
`RotRho` acting on commitments through the Ajtai S-module homomorphism.

- Norm cost: mixing multiplies witness norm by at most the expansion factor `T`, so the
  output claim has norm bound `B = b^k`. Parameter validity requires
  `(K+k)·T·(b−1) < B` (`paper/sampling.rs::check_rlc_bound`).
- The verifier recomputes the combined claim itself from the inputs and challenges; the
  prover's claimed output is never trusted (`paper/reductions/pi_rlc.rs`).

## Π_DEC — decomposition (§7.5)

Splits the norm-`B` claim back into `k` children of norm `b` via balanced b-ary
decomposition `(z_1, …, z_k) ← split_b(z)`. The verifier checks the recomposition
linearly on commitments and claimed evaluations:

```text
c   ?=  Σ_i b^{i−1} · c_i
y_j ?=  Σ_i b^{i−1} · y_{i,j}
```

(`paper/reductions/pi_dec.rs`). The children are the next accumulator: low-norm again,
so Ajtai binding holds for the next fold. This step is the lattice-specific part of the
pipeline — HyperNova over Pedersen commitments needs no norm control.

## The embedding trick (§5)

CCS arithmetic happens over `F_q`, but commitments live over the ring
`R_q = F_q[X]/Φ_81`. SuperNeo's §5 evaluation homomorphism makes the two interoperate:
the `bar(·)` lift satisfies `Mz = ct(bar(M)·z)` (Thm. 4), so field-level matrix-vector
products are recoverable from ring-level products, and linear combinations of
evaluations commute with the commitment's S-action (Thm. 5). In code:
`neo_math::superneo_bar_*`, enforced at the seam by `paper/reductions/pi_rlc.rs`;
formally cross-checked in `formal/superneo-lean`.

## Engines

`neo_reductions::api::FoldingMode` selects the implementation:

| Mode | Use |
|---|---|
| `Optimized` | Production and all normal tests |
| `PaperExact` (feature `paper-exact`) | O(2^ℓ) brute-force reference, correctness cross-check only |
| `OptimizedWithCrosscheck` (feature `paper-exact`) | Runs both and compares, for debugging |

Strong/weak interactive-reduction security (SuperNeo §6) is what justifies composing
the three reductions and applying Fiat-Shamir; see
[Transcript & digests](transcript-and-digests.md) for the binding discipline.

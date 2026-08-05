# neo-ajtai

Ajtai matrix commitment (module-SIS) — the lattice commitment underpinning everything
foldable. `#![forbid(unsafe_code)]`.

## The commitment

`L: F_q^{d×m} → C` via `c = cf(M·cf⁻¹(Z))` for a uniform `M ∈ R_q^{κ×m}`
(SuperNeo Defs. 9, 11–13):

- **S-homomorphic** — ring elements act on commitments through rotation matrices
  (`s_mul`, `s_lincomb`, `scale_commitment*`), which is what lets Π_RLC mix
  commitments by challenges without opening them.
- **(d, m, B)-binding** — binding holds only for openings with ℓ∞ norm below `B`;
  this is why Π_DEC must re-normalize the accumulator every fold.
- **Pay-per-bit** — commitment cost scales with witness bit-width (Sec. 3.2–3.3),
  the property the low-norm F′ bit-image layout exploits.

## API surface

- `setup` / `setup_par` — sample public parameters `PP`.
- `commit`, `commit_row_major`, `try_commit*`, masked/precomputed variants;
  `verify_open` / `verify_split_open`.
- `decomp_b` / `split_b` / `assert_range_b` (`DecompStyle`) — decomposition used by
  Π_DEC and the pay-per-bit embedding.
- `s_module::AjtaiSModule` — the `SModuleHomomorphism` implementation `neo-ccs`
  traits expect, plus the **global PP registry** (`set_global_pp*`,
  `get_global_pp_for_dims`, seeded variants): verifier-owned setup keyed by shape,
  so provers cannot supply their own parameters.
- Seeded/`#[doc(hidden)]` commit auditing helpers back deterministic test setups
  (`tests/seeded_signed_unit_commit_parity.rs` in this crate).

## Deliberate omission

The crate does **not** export `verify_linear` (compile-fail doctest enforces it):
linear-relation verification belongs to the folding layer (Π_RLC), not the commitment
crate — keeping a single owner for that soundness obligation.

Executable behavior checks live under `crates/neo-ajtai/tests/`. Protocol-critical
rules must cite the pinned paper or the active Lean model instead of a copied
per-crate specification.

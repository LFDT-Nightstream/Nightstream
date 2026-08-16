# Campaign profile v2 freeze (bar 2, amended)

Amendment (2026-08-16). Profile v1 (k_rho = 2) was frozen 2026-08-15 and
failed in practice: the paper's Definition-14 guard
`(K + k) * T * (b - 1) < B = b^k_rho` rejects every fold at k_rho = 2, so no
honest recursive assignment exists for v1 (named by the 6,142 s capture
diagnostic). The user directed that parameters follow the papers'
soundness and completeness requirements. Pi_DEC always re-enters
`k = k_rho` accumulator limbs into the next Pi_RLC
(`crates/neo-fold-clean/src/engine/optimized.rs`, `let k = pp.k_rho()`),
so this one-fresh-claim profile folds `count = k_rho + 1` claims per step
and the guard becomes `(k_rho + 1) * 216 * 1 < 2^k_rho`. The smallest
solution is `k_rho = 12` (13 * 216 = 2,808 < 4,096; k_rho = 11 gives
2,592 > 2,048). Every extra limb is one full-width committed column
block, so the guard minimum is also the committed-column minimum. Paper
B.2 uses k = 14 only to allow up to 61 fresh claims per fold.

The single owner of the frozen construction is
`bridge/src/campaign_profile.rs` (`campaign_profile_params`,
`campaign_profile_plan`, `campaign_profile_audit`); every campaign test
builds through it. The drift gate is
`campaign_profile_v2_digests_are_frozen` in
`bridge/tests/profile_freeze.rs`. `production_claim` stays `false`; the
production regime decision is still open at the protocol level, and final
certification re-runs when it lands.

## Construction

The two physical arms come from one exact Rust audit:

- Parameters: Goldilocks paper B.2 shape with `kappa = 1`, `k_rho = 12`,
  `t = 1`, `lambda = 1`; all other values from `neo_params::goldilocks_paper_b2`
  (`Params::test_only_from_neo_params`).
- Memory profile: `NebulaParams::new(0, 0, 1, 2, 1)`; ROM `[7]`.
- Plan: `NebulaPlan::new(memory, [7], [0xDA; 32], kappa)`.
- Preprocessing: `NebulaFPrimePreprocessing::new_seeded(params, plan, 0xDA00_0001)`.
- Audit: `NebulaFPrimeRelation::audit_fixed_point_constraint_sources`.

The terminal relation comes from the shared combined manifest fixture and
is independent of the arm parameters:

- Manifest: `combined_manifest()` from
  `crates/neo-fold-clean/tests/support/lean_manifest_fixture.rs`.
- Parameters: `Params::goldilocks_paper_b2()`; Ajtai log via `TEST_AJTAI_SEED`.
- 14 zero running claims and witnesses; honest fresh instance;
  `compile_combined_terminal_r1cs`.

## Pinned digests and geometry (v2)

| Relation | Value |
|---|---|
| Base arm source digest | `sha256:e5f31e44449fd9bdf41f742f0afd6a9cee93be2fe98b1dedfa4d27f6aa250570` |
| Base arm geometry | 39,949 rows; 38,626 columns; 2,426 public |
| Recursive arm source digest | `sha256:f06cd06435b8060f0c94adaddeb8349a24ba784b974a6bac7a06ca9e93163915` |
| Recursive arm geometry | 11,187,825 rows; 11,078,210 columns; 2,426 public |
| Selective fixed point (final plan) digest | `sha256:42eb7385d90b1de44cb67a505ae5ba1634559f105c315031acb681401449b965` |
| Selective fixed point geometry | 3,666,055 rows; 13,314,834 columns; 2,430 public |
| Terminal source digest | `sha256:85b400cebcfaa8fac702072aff342d67c6acca87e4470199d86a935c98264461` |
| Terminal source geometry | 58,593 rows; 58,592 columns; 48,871 public |
| Terminal diagnostic digest | `sha256:63664e95c3f91dcf35db99ad3e0dd235643d274e5ccfd9be6a18252eb8a12f98` |
| Terminal padded Spartan geometry | 65,536 rows; 114,407 columns |

Family counts: base 6, terminal 8, recursive 82. Recursive-arm encoding
measurements (compact pipeline compatibility): 51,072,145 explicit terms,
527 distinct coefficients, max 9,074 terms per row, 36 seeded blocks, no
geometric runs.

## Measured facts behind the pins

- The base arm keeps its v1 geometry but changes coefficients with k_rho;
  base classifications re-derive under v2.
- The terminal relation is byte-identical to v1 (paper-B.2 fixture); its
  8/8 certification survives the amendment.
- Source digests are plan-seed-independent (measured on v1, 0xDA vs 0xD9);
  the final plan digest binds to the 0xDA mirror shape.
- Profile names in generated artifacts stay
  `campaign-base-classification-v1`, `campaign-recursive-classification-v1`,
  `campaign-terminal-classification-v1` (the digest, not the label, is the
  authority; the labels identify the classification lineage).

## Profile v1 record (superseded)

k_rho = 2; base digest `sha256:54bec6fa...`, recursive
`sha256:4c0a5164...` (4,530,315 x 4,480,464), final plan
`sha256:3024cf0e...` (1,415,271 x 6,559,326). Unfoldable by Definition 14;
kept here as the amendment audit trail. Rejected earlier regimes:
lambda=125 paper B.2 (extension-policy census provides 114 bits), and
lambda=114 paper B.2 (audit construction alone exceeded 2 h 06 m).

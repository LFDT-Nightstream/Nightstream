# Campaign profile v1 freeze (bar 2)

Decision (user, 2026-08-15): freeze the campaign on the minimal shape and
classify against it. The production regime decision is still open at the
protocol level. When it lands, the classification re-runs against the chosen
production profile. `production_claim` stays `false` until that review. This
file pins the artifact digests. The drift gate is
`campaign_profile_v1_digests_are_frozen` in `bridge/tests/profile_freeze.rs`.

## Construction

The two physical arms come from one exact Rust audit:

- Parameters: Goldilocks paper B.2 shape with `kappa = 1`, `k_rho = 2`,
  `t = 1`, `lambda = 1`; all other values from `neo_params::goldilocks_paper_b2`
  (`Params::test_only_from_neo_params`).
- Memory profile: `NebulaParams::new(0, 0, 1, 2, 1)`; ROM `[7]`.
- Plan: `NebulaPlan::new(memory, [7], [0xDA; 32], kappa)`.
- Preprocessing: `NebulaFPrimePreprocessing::new_seeded(params, plan, 0xDA00_0001)`.
- Audit: `NebulaFPrimeRelation::audit_fixed_point_constraint_sources`.

The terminal relation comes from the shared combined manifest fixture:

- Manifest: `combined_manifest()` from
  `crates/neo-fold-clean/tests/support/lean_manifest_fixture.rs`.
- Parameters: `Params::goldilocks_paper_b2()`; Ajtai log via `TEST_AJTAI_SEED`.
- 14 zero running claims and witnesses; honest fresh instance;
  `compile_combined_terminal_r1cs`.

## Pinned digests and geometry

| Relation | Value |
|---|---|
| Base arm source digest | `sha256:54bec6fa7de4ec475e2fd43a1c015bfede809d2d1370b67677ea66dbda6839e7` |
| Base arm geometry | 39,949 rows; 38,626 columns; 2,426 public |
| Recursive arm source digest | `sha256:4c0a51647877cd072970c160d49d1dc78b7d34b39dd3e7613c716cef2869934e` |
| Recursive arm geometry | 4,530,315 rows; 4,480,464 columns; 2,426 public |
| Selective fixed point (final plan) digest | `sha256:3024cf0eea6ac9093157e5dc1674187abc9fa3f17f8598d72ab41e45504e50fc` |
| Selective fixed point geometry | 1,415,271 rows; 6,559,326 columns; 2,430 public |
| Terminal source digest | `sha256:85b400cebcfaa8fac702072aff342d67c6acca87e4470199d86a935c98264461` |
| Terminal source geometry | 58,593 rows; 58,592 columns; 48,871 public |
| Terminal diagnostic digest | `sha256:63664e95c3f91dcf35db99ad3e0dd235643d274e5ccfd9be6a18252eb8a12f98` |
| Terminal padded Spartan geometry | 65,536 rows; 114,407 columns |

Family counts: base 6, terminal 8, recursive 82.

## Measured facts behind the pins

- The source artifact digests do not depend on the plan seed. The `0xDA` and
  `0xD9` plans produce the same base and recursive source digests
  (`print_campaign_profile_v1_digests`, 2026-08-15). Classifications that
  build audits with either seed bind to the same source matrices.
- The final plan digest does depend on the plan seed. The frozen value is the
  `0xDA` mirror shape. The committed Lean mirrors
  (`Generated/BaseBoundArtifact.lean`, `Generated/TerminalBoundArtifact.lean`)
  embed the same digests as this table.
- Profile names in generated artifacts: `campaign-base-classification-v1`,
  `campaign-recursive-classification-v1`, `campaign-terminal-classification-v1`.

## Rejected alternatives (evidence in CAMPAIGN.md)

- Paper B.2 with `lambda = 125`: rejected by the verifier's own
  extension-policy census; the shape provides 114 bits.
- Paper B.2 with `lambda = 114`: the audit construction alone exceeded
  2 h 06 m; impractical for iterated campaign use. The minimal shape builds
  its audit in seconds and its complete base export in under 2 s.

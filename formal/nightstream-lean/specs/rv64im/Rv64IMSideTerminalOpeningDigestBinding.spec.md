# Rv64IMSideTerminalOpeningDigestBinding Spec

## Purpose

- **What it is**: The Nightstream RV64IM theorem boundary for the side-terminal opening-artifact digest.
- **What it is not**: It is not a proof of the full Nightstream verifier, the side-terminal backend SNARK, or the accepted-artifact replay layer.
- **Protocol role**: It states when the side-terminal theorem may carry an opening-artifact digest and when Rust is allowed to treat that carried digest as theorem-bound.

## Contract

The owner formalizes three views of the same boundary:

- the legacy native side-terminal check shape, which accepts a claim-side witness, an opening-side witness, and a locally consistent digest chain without requiring the carried `opening_artifact_digest` to equal the canonical digest entailed by the accepted artifact,
- the missing semantic condition that the carried `opening_artifact_digest` must equal the canonical digest determined by the accepted artifact or the theorem-owned opening-artifact constructor,
- the corrected side-terminal boundary, which keeps the native claim/opening witness checks and also requires canonical opening-digest binding.

For every exported case, the owner must expose:

- whether the locally carried native side-terminal witness checks pass,
- whether the carried opening-artifact digest equals the canonical opening-artifact digest,
- whether the corrected boundary accepts the case,
- whether the observed Rust result refines the corrected boundary,
- whether the case is an explicit counterexample showing that the legacy native shape accepts while the corrected boundary rejects,
- and the exact blocker strings induced by any case that does not refine the corrected boundary.

## Check Targets

- `currentSideTerminalCheck`
- `canonicalOpeningDigestBound`
- `fixedSideTerminalCheck`
- `rustRefinesFixedSideTerminalCheck`
- `sideTerminalOpeningDigestBindingReport`
- `rv64imSideTerminalOpeningDigestBindingReports`
- `validGeneratedRv64imSideTerminalOpeningDigestBindingCases`
- `sideTerminalOpeningDigestBindingCounterexamples`
- `uniqueSideTerminalOpeningDigestBindingBlockers`

## Dependency and Consumer Map

- **Depends on**:
  - `Nightstream/Rv64IM/Generated/SideTerminalOpeningDigestBindingCorpus.lean`
- **Consumed by**:
  - `Nightstream/Rv64IM/ProofCompleteAudit.lean`
  - `Nightstream/Rv64IM/ProofCompleteAuditInterface.lean`
  - `Nightstream.lean`

## Out of Scope

- re-proving the side-terminal backend shell
- reconstructing accepted artifacts from Rust proof objects
- proving the full Nightstream verifier end to end

import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Polynomial.PackedRows
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentitySecurity

/-!
Contract: production Goldilocks specialization of centered-domain row packing.

Owns: discharge of the projective-seven premise for the exact two-coordinate
row and its fixed-zero odd tail.

Does not own: generated matrix coefficients, row placement, selector
Booleanity, Rust conformance, row multiplicity, or permission to remove rows.

Emits constraints: no.

Assurance tier: security-reduced for the Goldilocks nonresidue premise and
model-level for row semantics. This module is not artifact-checked or
Rust-conformant.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CenteredDomainPacking

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Components
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Semantics

/-- One active packed row is zero exactly when both centered-unit residuals
are zero over the production Goldilocks field. -/
theorem production_centeredPair_zero_iff (left right : F) :
    evaluate (centeredPairPoint 1 left right) = 0 ↔
      centeredUnitResidual left = 0 ∧ centeredUnitResidual right = 0 :=
  Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.PackedRows.evaluate_centeredPairPoint_one_zero_iff
    Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentitySecurity.sevenProjectiveNonresidue
    left right

/-- A fixed-zero odd tail is zero exactly when its live centered-unit
residual is zero over the production Goldilocks field. -/
theorem production_centeredTail_zero_iff (left : F) :
    evaluate (centeredPairPoint 1 left 0) = 0 ↔
      centeredUnitResidual left = 0 :=
  Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.PackedRows.evaluate_centeredPairTailPoint_one_zero_iff
    Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentitySecurity.sevenProjectiveNonresidue
    left

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CenteredDomainPacking

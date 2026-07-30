import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcPointBinding

/-!
Focused mutation regression for the shared `Pi_CCS`/`Pi_RLC` point carrier.

Changing even the first physical column of a realized extension coordinate
cannot preserve the carried expression.  This isolates the exact substitution
that a free or copied `Pi_RLC` point would permit.
-/

set_option autoImplicit false

namespace NightstreamTests.PaperNifsPiRlcPointBindingMutation

open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open PaperNifsPiRlcPointBinding

def changedLowColumn (value : Carried) : Carried :=
  KTraceProgram.decodePoint
    { c0 := (carriedColumns value).c0 + 1
      c1 := (carriedColumns value).c1 }

/-- A substituted low coordinate is not the physical pair extracted from the
selected transcript replay. -/
theorem changedLowColumn_ne_realized
    (value : Carried) (realized : RealizesColumns value) :
    changedLowColumn value ≠ value := by
  intro equal
  rw [realized] at equal
  have lowEqual := congrArg (fun carried : Carried => carried.low) equal
  simp [changedLowColumn, KTraceProgram.decodePoint, carriedColumns,
    firstColumn] at lowEqual

end NightstreamTests.PaperNifsPiRlcPointBindingMutation

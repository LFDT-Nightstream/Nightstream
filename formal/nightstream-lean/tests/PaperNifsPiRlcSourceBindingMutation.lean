import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcSourceBinding

/-!
Mutation regression for the public `Pi_RLC` source carrier.

The negative control substitutes one numeric column in the first commitment
coefficient while preserving every list width.  The decoded opening changes,
so width-only or caller-supplied source equalities cannot replace the
selected-frame column binding.
-/

set_option autoImplicit false

namespace NightstreamTests.PaperNifsPiRlcSourceBindingMutation

open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc

def zeroColumns : ProjectionColumns 1 where
  commitment _ := List.replicate 54 0
  x _ := List.replicate 54 0
  yRing _ _ := List.replicate 54 0

def substitutedColumns : ProjectionColumns 1 where
  commitment lane :=
    if lane.val = 0 then 1 :: List.replicate 53 0
    else List.replicate 54 0
  x _ := List.replicate 54 0
  yRing _ _ := List.replicate 54 0

def distinguishingAssignment (column : Nat) : Nat :=
  if column = 1 then 1 else 0

/-- A single source-column substitution is observable even though both
carriers retain the required coefficient width. -/
theorem single_column_substitution_changes_decoded_source :
    decodeOpening distinguishingAssignment substitutedColumns ≠
      decodeOpening distinguishingAssignment zeroColumns := by
  intro equal
  have selected := congrArg
    (fun opening =>
      (opening.commitment (0 : Fin 18)).getD 0 0)
    equal
  simp [decodeOpening, substitutedColumns, zeroColumns,
    values, distinguishingAssignment, residue] at selected
  have modulus_ne_one :
      Nightstream.Implementation.R1CS.goldilocksP ≠ 1 := by
    decide
  exact modulus_ne_one selected

end NightstreamTests.PaperNifsPiRlcSourceBindingMutation

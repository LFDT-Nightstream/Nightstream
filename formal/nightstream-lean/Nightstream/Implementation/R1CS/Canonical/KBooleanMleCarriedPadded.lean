import Nightstream.Implementation.R1CS.Canonical.KBooleanMleOwnership
import Nightstream.Implementation.R1CS.Canonical.KBooleanMlePadded

/-!
Contract: physical zero-extension of the 54 authoritative Phi81 lanes to a
complete Boolean table.

Only the live `Fin ringDegree` values are caller-owned.  Every other leaf is
the row-free canonical zero, so a prover cannot supply padded lane values.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KBooleanMleCarriedPadded

open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial

/-- Zero-extend the exact 54 carried source lanes to a complete Boolean cube. -/
def carriedTable
    {variables : Nat}
    (values : Fin ringDegree → Carried) :
    BooleanTable Carried variables :=
  BooleanTable.tabulate fun vertex =>
    if bounded : NumericBooleanDomain.index vertex < ringDegree then
      values ⟨NumericBooleanDomain.index vertex, bounded⟩
    else
      KLinear.zeroCarried

private theorem decodeTable_tabulate
    (assignment : Nat → Nat) :
    ∀ {variables : Nat}
      (values : BooleanVertex variables → Carried),
      KBooleanMleSemantics.decodeTable assignment
          (BooleanTable.tabulate values) =
        BooleanTable.tabulate
          (fun vertex =>
            KBooleanMleSemantics.decodeCarried assignment (values vertex))
  | 0, values => rfl
  | variables + 1, values => by
      simp only [BooleanTable.tabulate, KBooleanMleSemantics.decodeTable]
      rw [decodeTable_tabulate assignment
          (fun tail => values (.cons false tail)),
        decodeTable_tabulate assignment
          (fun tail => values (.cons true tail))]

@[simp] theorem decodeCarried_zero (assignment : Nat → Nat) :
    KBooleanMleSemantics.decodeCarried assignment KLinear.zeroCarried =
      K.zero := by
  rfl

/-- Decoding the physical carried table yields exactly the semantic padded
table used by the unchanged FE formula. -/
theorem decodeTable_carriedTable
    (assignment : Nat → Nat)
    {variables : Nat}
    (values : Fin ringDegree → Carried) :
    KBooleanMleSemantics.decodeTable assignment
        (carriedTable (variables := variables) values) =
      KBooleanMlePadded.semanticTable
        (fun lane => KBooleanMleSemantics.decodeCarried assignment (values lane)) := by
  rw [carriedTable, decodeTable_tabulate]
  unfold KBooleanMlePadded.semanticTable
  apply congrArg BooleanTable.tabulate
  funext vertex
  by_cases bounded : NumericBooleanDomain.index vertex < ringDegree
  · simp [bounded]
  · simp [bounded]

/-- A satisfying physical MLE over the derived padded table computes the exact
54-lane semantic evaluation. -/
theorem rows_compute_paddedLaneEvaluation
    (assignment : Nat → Nat)
    (base : Nat)
    {domain : FlatNcDomain}
    (covers : Fe.LaneCovers domain)
    (values : Fin ringDegree → Carried)
    (coordinates : List Carried)
    (dimension : coordinates.length = domain.laneVariables)
    (satisfied :
      Satisfies
        (KBooleanMle.rows (KFrames.frameAt base)
          (carriedTable (variables := domain.laneVariables) values)
          coordinates 0)
        assignment) :
    KBooleanMleSemantics.decodeCarried assignment
        (KBooleanMle.carried (KFrames.frameAt base)
          (carriedTable (variables := domain.laneVariables) values)
          coordinates 0) =
      Fe.paddedLaneEvaluation covers
        (fun lane =>
          KBooleanMleSemantics.decodeCarried assignment (values lane))
        (KBooleanMleSemantics.decodePoint assignment coordinates dimension) := by
  apply KConcreteBridge.ofConcrete_injective
  rw [KBooleanMleSemantics.ofConcrete_decodeCarried]
  rw [KBooleanMleSemantics.rows_compute_evaluate
    assignment base
    (carriedTable (variables := domain.laneVariables) values)
    coordinates dimension satisfied]
  rw [decodeTable_carriedTable]
  rw [KBooleanMlePadded.semanticTable_evaluate covers]

end Nightstream.Implementation.R1CS.Canonical.KBooleanMleCarriedPadded

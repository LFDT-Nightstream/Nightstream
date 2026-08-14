import Nightstream.Implementation.Nebula.Production.FPrime.Terminal.RunningPlacementFor

/-!
Contract: hostile countermodel for omitted final-running carrier placement.

Exact coordinate aliases do not establish the values stored in their source
carrier. If the final NIFS producer does not bind the complete carrier, a
terminal consumer can read a uniformly shifted value instead of the canonical
running encoding.

Assurance tier: model-level negative result.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.TerminalRunningPlacementCountermodels

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductionPaperTerminalRunningPlacementFor
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

def collapsedColumn : ColumnId :=
  { owner := .prelude, bundleIndex := 0, coordinateIndex := 0 }

def collapsedCarrier (rowVariables : Nat) : Carrier rowVariables where
  column := fun _ => collapsedColumn

noncomputable def shiftedAssignment
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : Running rowVariables logicalWidth publicFits) : ColumnId -> F :=
  fun _ =>
    ((ProductNifsCodec.runningCodecFor rowVariables
      (FullShape rowVariables logicalWidth publicFits)).encode running).getD
        0 0 + 1

/-- Without `Placed`, a physical carrier can disagree with the canonical
running codec at its first coordinate. Thus aliases alone cannot authorize the
terminal verifier input. -/
theorem omitted_placement_allows_wrong_carrier
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : Running rowVariables logicalWidth publicFits) :
    ¬ Placed (collapsedCarrier rowVariables)
      (shiftedAssignment running) running := by
  intro placed
  let first : Fin (ProductNifsCodec.runningFieldCountFor rowVariables) :=
    ⟨0, by simp [ProductNifsCodec.runningFieldCountFor]⟩
  have firstExact := placed.coordinate first
  change
    ((ProductNifsCodec.runningCodecFor rowVariables
      (FullShape rowVariables logicalWidth publicFits)).encode running).getD
          0 0 + 1 =
      ((ProductNifsCodec.runningCodecFor rowVariables
        (FullShape rowVariables logicalWidth publicFits)).encode running).getD
          0 0 at firstExact
  have impossible := congrArg (fun value => value -
    ((ProductNifsCodec.runningCodecFor rowVariables
      (FullShape rowVariables logicalWidth publicFits)).encode running).getD
        0 0) firstExact
  norm_num [goldilocksModulus] at impossible

end Nightstream.Implementation.Nebula.TerminalRunningPlacementCountermodels

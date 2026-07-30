import Nightstream.Implementation.R1CS.Canonical.KPiCcsTranscript
import Nightstream.Implementation.R1CS.Canonical.KTraceProgram
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc

/-!
Contract: reuse the exact physical `Pi_CCS` SumCheck-point squeeze as the
shared public point of the following `Pi_RLC` occurrence.

The transcript already materializes every extension coordinate as two
Poseidon2 output columns.  This module extracts those columns from the
singleton carried expressions and proves that their ordinary point decoder
is the unchanged `Pi_CCS` decoded point.  It emits no copy rows and allocates
no replacement point.

Assurance tier: model-level canonical encoding.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcPointBinding

open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc

/-- First referenced column, used only after `RealizesColumns` proves that the
carried expression is a unit singleton. -/
def firstColumn (terms : LinCombNormal.LinComb) : Nat :=
  (terms.getD 0 (0, 0)).1

/-- The two physical columns named by a singleton carried value. -/
def carriedColumns (value : Carried) :
    Nightstream.Implementation.R1CS.ProjectionProgram.KColumns where
  c0 := firstColumn value.low
  c1 := firstColumn value.high

/-- Exact condition under which `carriedColumns` is an extraction rather than
a defaulting decoder. -/
def RealizesColumns (value : Carried) : Prop :=
  value = KTraceProgram.decodePoint (carriedColumns value)

/-- Every value returned by the duplex extension squeeze names its two output
ports as unit singletons. -/
theorem squeezeK_realizesColumns
    (base : Nat) (builder : SymbolicDuplex.Builder) :
    RealizesColumns (SymbolicDuplex.squeezeK base builder).1 := by
  unfold RealizesColumns SymbolicDuplex.squeezeK SymbolicDuplex.gate
    SymbolicDuplex.permute SymbolicDuplex.outputState carriedColumns
    firstColumn KTraceProgram.decodePoint
  rfl

/-- Every result of one indexed squeeze run is backed by physical output
columns. -/
theorem squeezeIndexedGo_realizesColumns
    (base label index remaining : Nat)
    (builder : SymbolicDuplex.Builder) :
    ∀ value ∈
      (KPiCcsTranscript.squeezeIndexedGo
        base label index remaining builder).1,
      RealizesColumns value := by
  induction remaining generalizing index builder with
  | zero =>
      intro value member
      cases member
  | succ remaining inductionHypothesis =>
      simp only [KPiCcsTranscript.squeezeIndexedGo]
      intro value member
      rcases List.mem_cons.mp member with head | tail
      · subst value
        exact squeezeK_realizesColumns _ _
      · exact inductionHypothesis _ _ value tail

/-- Every SumCheck challenge in the causal replay is an exact physical pair. -/
theorem replayRounds_realizesColumns
    {degree : Nat} (base : Nat)
    (rounds : List (KFixedPhaseSumCheck.Round degree))
    (index : Nat) (builder : SymbolicDuplex.Builder) :
    ∀ value ∈
      (KPiCcsTranscript.replayRounds base rounds index builder).challenges,
      RealizesColumns value := by
  induction rounds generalizing index builder with
  | nil =>
      intro value member
      cases member
  | cons round rest inductionHypothesis =>
      simp only [KPiCcsTranscript.replayRounds]
      intro value member
      rcases List.mem_append.mp member with head | tail
      · exact squeezeIndexedGo_realizesColumns _ _ _ _ _ value head
      · exact inductionHypothesis _ _ value tail

/-- The complete selected replay's point list contains only exact physical
pairs. -/
theorem replay_point_realizesColumns
    {shape : Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Shape}
    {degree : Nat}
    (input : KPiCcsTranscript.Input shape degree) :
    ∀ value ∈ (KPiCcsTranscript.replay input).point,
      RealizesColumns value := by
  exact replayRounds_realizesColumns _ _ _ _

/-- Physical point columns in the exact replay order. -/
def pointColumns
    {shape : Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Shape}
    {degree : Nat}
    (input : KPiCcsTranscript.Input shape degree) : PointColumns where
  r := (KPiCcsTranscript.replay input).point.map fun value =>
    let columns := carriedColumns value
    (columns.c0, columns.c1)

theorem decoded_realizedColumns
    (assignment : Nat → Nat) (value : Carried)
    (realized : RealizesColumns value) :
    KPointEquality.decoded assignment value =
      Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.extensionValue
        assignment
        ((carriedColumns value).c0, (carriedColumns value).c1) := by
  rw [realized]
  apply
    Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.k_eq_of_coeffs
  · simp [KPointEquality.decoded, KTraceProgram.decodePoint,
      carriedColumns, firstColumn, KFixedPhaseSumCheck.decodeCarried,
      KConcreteFixedPhaseBridge.ofProjection,
      Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.extensionValue,
      Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.residue,
      KMul.lcEval_singleton_col]
  · simp [KPointEquality.decoded, KTraceProgram.decodePoint,
      carriedColumns, firstColumn, KFixedPhaseSumCheck.decodeCarried,
      KConcreteFixedPhaseBridge.ofProjection,
      Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.extensionValue,
      Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.residue,
      KMul.lcEval_singleton_col]

/-- Reading the replay point through its exact finite-function view recovers
the original ordered list. -/
theorem ofFn_pointAt_eq_point
    {shape : Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Shape}
    (execution : KPiCcsTranscript.Replay shape) :
    List.ofFn (KPiCcsTranscript.pointAt execution) = execution.point := by
  apply List.ext_get
  · simp [execution.point_length]
  · intro index leftLt rightLt
    simp only [List.get_eq_getElem, List.getElem_ofFn]
    rfl

/-- The point consumed by the `Pi_RLC` source is definitionally the same
Fiat–Shamir point decoded by the preceding `Pi_CCS` occurrence. -/
theorem decodePointColumns_eq_piCcsPoint
    {shape : Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Shape}
    {degree : Nat}
    (input : KPiCcsTranscript.Input shape degree)
    (assignment : Nat → Nat) :
    decodePointColumns assignment (pointColumns input) =
      (KPiCcsOccurrence.decodedPoint
        (KPiCcsTranscript.occurrenceInput input) assignment).coordinates := by
  calc
    decodePointColumns assignment (pointColumns input) =
        (KPiCcsTranscript.replay input).point.map
          (KPointEquality.decoded assignment) := by
      unfold decodePointColumns pointColumns
        Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.extensionValues
      simp only [List.map_map]
      apply List.map_congr_left
      intro value member
      exact (decoded_realizedColumns assignment value
        (replay_point_realizesColumns input value member)).symm
    _ = (KPiCcsOccurrence.decodedPoint
          (KPiCcsTranscript.occurrenceInput input) assignment).coordinates := by
      rw [← ofFn_pointAt_eq_point (KPiCcsTranscript.replay input)]
      unfold KPiCcsOccurrence.decodedPoint KPiCcsTerminal.decodedPoint
        KPiCcsTerminal.alphaEqualityInput KPointEquality.decodedLeft
        KPointEquality.indices KPiCcsOccurrence.terminalInput
        KPiCcsTranscript.occurrenceInput
      simp only [List.map_ofFn]
      rfl

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcPointBinding

import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge

/-!
Structural validation for the deterministic derived-column program used by
the bounded selective `y_zcol` artifact.

Owns: the previous-column/fresh-output invariant, its executable checker,
the checker-to-kernel-proof bridge, and the five coherent program partitions
used to keep artifact checking inside the Lean memory bound.

Does not own: honest source semantics, field execution, selected-row
satisfaction, projection authority, or security reduction.

Emits constraints: no.

| Structural leaf | Mathematical obligation | Authority class |
|---|---|---|
| checker bridge | previous inputs are known and derived outputs are fresh | derived |
| program partition | five chunks cover the canonical rewrite program exactly | computed |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge

def PreviousKnown (known : List Nat) (step : DecodedRewriteStep) : Prop :=
  match step.previous with
  | none => True
  | some slot => slot.compilerIndex ∈ known

private instance (known : List Nat) (step : DecodedRewriteStep) :
    Decidable (PreviousKnown known step) := by
  unfold PreviousKnown
  cases step.previous <;> infer_instance

inductive DerivedWellFormed : List Nat → List DecodedRewriteStep → Prop
  | nil (known) : DerivedWellFormed known []
  | source {known step rest output}
      (previous : PreviousKnown known step)
      (isSource : step.output = .source output)
      (tail : DerivedWellFormed known rest) :
      DerivedWellFormed known (step :: rest)
  | derived {known step rest slot}
      (previous : PreviousKnown known step)
      (isDerived : step.output = .derivedProductSum slot)
      (fresh : slot.compilerIndex ∉ known)
      (tail : DerivedWellFormed (slot.compilerIndex :: known) rest) :
      DerivedWellFormed known (step :: rest)

structure DerivedStepShape where
  previous : Option Nat
  output : Option Nat
deriving DecidableEq, Repr

/-- Proof-free projection used by the bounded artifact certificates. `none`
at `output` denotes a source output; `some column` denotes a fresh derived
output. -/
def derivedStepShape (step : DecodedRewriteStep) : DerivedStepShape :=
  { previous := step.previous.map (fun slot => slot.compilerIndex)
    output :=
      match step.output with
      | .source _ => none
      | .derivedProductSum slot => some slot.compilerIndex }

def ShapePreviousKnown (known : List Nat) (shape : DerivedStepShape) : Prop :=
  match shape.previous with
  | none => True
  | some column => column ∈ known

private instance (known : List Nat) (shape : DerivedStepShape) :
    Decidable (ShapePreviousKnown known shape) := by
  unfold ShapePreviousKnown
  cases shape.previous <;> infer_instance

private theorem shapePreviousKnown_iff
    (known : List Nat) (step : DecodedRewriteStep) :
    ShapePreviousKnown known (derivedStepShape step) ↔
      PreviousKnown known step := by
  cases previousEq : step.previous with
  | none =>
      simp [ShapePreviousKnown, PreviousKnown, derivedStepShape, previousEq]
  | some slot =>
      simp [ShapePreviousKnown, PreviousKnown, derivedStepShape, previousEq]

def derivedShapeWellFormedCheck :
    List Nat → List DerivedStepShape → Bool
  | _, [] => true
  | known, shape :: rest =>
      decide (ShapePreviousKnown known shape) &&
        match shape.output with
        | none => derivedShapeWellFormedCheck known rest
        | some column =>
            decide (column ∉ known) &&
              derivedShapeWellFormedCheck (column :: known) rest

/-- Generic kernel bridge from a compact shape certificate to the original
typed rewrite program. The executable check never inspects proof fields. -/
theorem derivedWellFormed_of_shape_check_true :
    ∀ known steps,
      derivedShapeWellFormedCheck known (steps.map derivedStepShape) = true →
        DerivedWellFormed known steps := by
  intro known steps
  induction steps generalizing known with
  | nil =>
      intro _
      exact .nil known
  | cons step rest inductionHypothesis =>
      intro checked
      rw [List.map_cons, derivedShapeWellFormedCheck] at checked
      have checkedParts :
          decide (ShapePreviousKnown known (derivedStepShape step)) = true ∧
            (match (derivedStepShape step).output with
              | none =>
                  derivedShapeWellFormedCheck known
                    (rest.map derivedStepShape)
              | some column =>
                  decide (column ∉ known) &&
                    derivedShapeWellFormedCheck (column :: known)
                      (rest.map derivedStepShape)) = true := by
        simpa only [Bool.and_eq_true] using checked
      rcases checkedParts with ⟨previousCheck, tailCheck⟩
      have previous : PreviousKnown known step :=
        (shapePreviousKnown_iff known step).mp
          (of_decide_eq_true previousCheck)
      cases outputEq : step.output with
      | source output =>
          simp only [derivedStepShape, outputEq] at tailCheck
          exact .source previous outputEq
            (inductionHypothesis known tailCheck)
      | derivedProductSum slot =>
          simp only [derivedStepShape, outputEq, Bool.and_eq_true] at tailCheck
          exact .derived previous outputEq
            (of_decide_eq_true tailCheck.1)
            (inductionHypothesis (slot.compilerIndex :: known) tailCheck.2)

def knownAfterDerived :
    List Nat → List DecodedRewriteStep → List Nat
  | known, [] => known
  | known, step :: rest =>
      match step.output with
      | .source _ => knownAfterDerived known rest
      | .derivedProductSum slot =>
          knownAfterDerived (slot.compilerIndex :: known) rest

theorem derivedWellFormed_append
    {known : List Nat} {left right : List DecodedRewriteStep}
    (leftValid : DerivedWellFormed known left)
    (rightValid : DerivedWellFormed (knownAfterDerived known left) right) :
    DerivedWellFormed known (left ++ right) := by
  induction leftValid with
  | nil =>
      simpa [knownAfterDerived] using rightValid
  | source previous isSource tail inductionHypothesis =>
      apply DerivedWellFormed.source previous isSource
      apply inductionHypothesis
      simpa [knownAfterDerived, isSource] using rightValid
  | derived previous isDerived fresh tail inductionHypothesis =>
      apply DerivedWellFormed.derived previous isDerived fresh
      apply inductionHypothesis
      simpa [knownAfterDerived, isDerived] using rightValid

private abbrev chunkSize : Nat := 250

theorem chunkSize_le_certificateLimit : chunkSize ≤ 256 := by
  decide

def derivedProgramTail0 : List DecodedRewriteStep :=
  decodedRewriteSteps

def derivedProgramChunk0 : List DecodedRewriteStep :=
  derivedProgramTail0.take chunkSize

def derivedProgramTail1 : List DecodedRewriteStep :=
  derivedProgramTail0.drop chunkSize

def derivedProgramChunk1 : List DecodedRewriteStep :=
  derivedProgramTail1.take chunkSize

def derivedProgramTail2 : List DecodedRewriteStep :=
  derivedProgramTail1.drop chunkSize

def derivedProgramChunk2 : List DecodedRewriteStep :=
  derivedProgramTail2.take chunkSize

def derivedProgramTail3 : List DecodedRewriteStep :=
  derivedProgramTail2.drop chunkSize

def derivedProgramChunk3 : List DecodedRewriteStep :=
  derivedProgramTail3.take chunkSize

def derivedProgramTail4 : List DecodedRewriteStep :=
  derivedProgramTail3.drop chunkSize

def derivedProgramChunk4 : List DecodedRewriteStep :=
  derivedProgramTail4

def derivedProgramShapeChunk0 : List DerivedStepShape :=
  derivedProgramChunk0.map derivedStepShape

def derivedProgramShapeChunk1 : List DerivedStepShape :=
  derivedProgramChunk1.map derivedStepShape

def derivedProgramShapeChunk2 : List DerivedStepShape :=
  derivedProgramChunk2.map derivedStepShape

def derivedProgramShapeChunk3 : List DerivedStepShape :=
  derivedProgramChunk3.map derivedStepShape

def derivedProgramShapeChunk4 : List DerivedStepShape :=
  derivedProgramChunk4.map derivedStepShape

def derivedProgramKnown0 : List Nat := []

def derivedProgramKnown1 : List Nat :=
  knownAfterDerived derivedProgramKnown0 derivedProgramChunk0

def derivedProgramKnown2 : List Nat :=
  knownAfterDerived derivedProgramKnown1 derivedProgramChunk1

def derivedProgramKnown3 : List Nat :=
  knownAfterDerived derivedProgramKnown2 derivedProgramChunk2

def derivedProgramKnown4 : List Nat :=
  knownAfterDerived derivedProgramKnown3 derivedProgramChunk3

theorem decodedRewriteStepsLengthExact :
    decodedRewriteSteps.length = 1250 := by
  calc
    decodedRewriteSteps.length = rewritePairs.length := by
      simpa only [List.length_map] using
        (congrArg List.length rewritePairStepsExact).symm
    _ = Materialized.Artifact.rewriteRows.length := by
      simpa only [List.length_map] using
        congrArg List.length rewritePairRowsExact
    _ = 1250 := Materialized.Artifact.rewriteRowCount

/-- Half-open source positions owned by the five existing data-certificate
chunks. Pairwise ordering is stronger than non-overlap. -/
def derivedProgramChunkRanges : List (Nat × Nat) :=
  [(0, 250), (250, 500), (500, 750), (750, 1000), (1000, 1250)]

theorem derivedProgramChunkRangesOrdered :
    derivedProgramChunkRanges.Pairwise
      (fun left right => left.2 ≤ right.1) := by
  decide

theorem derivedProgramChunkLengthsExact :
    derivedProgramChunk0.length = 250 ∧
      derivedProgramChunk1.length = 250 ∧
      derivedProgramChunk2.length = 250 ∧
      derivedProgramChunk3.length = 250 ∧
      derivedProgramChunk4.length = 250 := by
  simp [derivedProgramChunk0, derivedProgramChunk1,
    derivedProgramChunk2, derivedProgramChunk3, derivedProgramChunk4,
    derivedProgramTail0, derivedProgramTail1, derivedProgramTail2,
    derivedProgramTail3, derivedProgramTail4, chunkSize,
    decodedRewriteStepsLengthExact]

theorem derivedProgramShapeChunkLengthsExact :
    derivedProgramShapeChunk0.length = 250 ∧
      derivedProgramShapeChunk1.length = 250 ∧
      derivedProgramShapeChunk2.length = 250 ∧
      derivedProgramShapeChunk3.length = 250 ∧
      derivedProgramShapeChunk4.length = 250 := by
  simpa only [derivedProgramShapeChunk0, derivedProgramShapeChunk1,
    derivedProgramShapeChunk2, derivedProgramShapeChunk3,
    derivedProgramShapeChunk4, List.length_map] using
    derivedProgramChunkLengthsExact

theorem derivedProgramChunksExact :
    derivedProgramChunk0 ++
        (derivedProgramChunk1 ++
          (derivedProgramChunk2 ++
            (derivedProgramChunk3 ++ derivedProgramChunk4))) =
      decodedRewriteSteps := by
  have split3 :
      derivedProgramChunk3 ++ derivedProgramChunk4 =
        derivedProgramTail3 := by
    exact List.take_append_drop chunkSize derivedProgramTail3
  have split2 :
      derivedProgramChunk2 ++ derivedProgramTail3 =
        derivedProgramTail2 := by
    exact List.take_append_drop chunkSize derivedProgramTail2
  have split1 :
      derivedProgramChunk1 ++ derivedProgramTail2 =
        derivedProgramTail1 := by
    exact List.take_append_drop chunkSize derivedProgramTail1
  have split0 :
      derivedProgramChunk0 ++ derivedProgramTail1 =
        derivedProgramTail0 := by
    exact List.take_append_drop chunkSize derivedProgramTail0
  simp only [split3, split2, split1, split0, derivedProgramTail0]

theorem derivedProgramShapeChunksExact :
    derivedProgramShapeChunk0 ++
        (derivedProgramShapeChunk1 ++
          (derivedProgramShapeChunk2 ++
            (derivedProgramShapeChunk3 ++ derivedProgramShapeChunk4))) =
      decodedRewriteSteps.map derivedStepShape := by
  simpa only [derivedProgramShapeChunk0, derivedProgramShapeChunk1,
    derivedProgramShapeChunk2, derivedProgramShapeChunk3,
    derivedProgramShapeChunk4, List.map_append] using
    congrArg (List.map derivedStepShape) derivedProgramChunksExact

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment

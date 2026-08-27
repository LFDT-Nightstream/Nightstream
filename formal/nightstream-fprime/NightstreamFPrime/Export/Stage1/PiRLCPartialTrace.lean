import NightstreamFPrime.Export.Stage1.PiRLCNonzero

/-!
Owns the exact indexed PiRLC combination trace used by executable parity.

Each step materializes its result in a proof-sized array before the next
source reads it. This preserves the left-to-right circuit recurrence and
prevents native evaluation from rebuilding every earlier prefix. The final
trace entries are proved equal to the canonical head-first paper folds.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCPartialTrace

open NightstreamFPrime.Export.Stage1.PiRLCNonzero
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism

/-- An array whose exact size is part of its value. -/
structure FixedArray (Alpha : Type) (count : Nat) where
  values : Array Alpha
  size_eq : values.size = count

namespace FixedArray

def ofFn {Alpha : Type} {count : Nat} (value : Fin count → Alpha) :
    FixedArray Alpha count where
  values := Array.ofFn value
  size_eq := by simp

def get {Alpha : Type} {count : Nat} (values : FixedArray Alpha count)
    (index : Fin count) : Alpha :=
  values.values[index.val]'(by
    rw [values.size_eq]
    exact index.isLt)

@[simp] theorem get_ofFn {Alpha : Type} {count : Nat}
    (value : Fin count → Alpha) (index : Fin count) :
    (ofFn value).get index = value index := by
  simp [ofFn, get]

def toList {Alpha : Type} {count : Nat}
    (values : FixedArray Alpha count) : List Alpha :=
  values.values.toList

@[simp] theorem toList_length {Alpha : Type} {count : Nat}
    (values : FixedArray Alpha count) : values.toList.length = count := by
  simp [toList, values.size_eq]

end FixedArray

abbrev MaterializedRingF := FixedArray F ringDegree
abbrev MaterializedRingK := FixedArray K ringDegree
abbrev MaterializedCommitment :=
  FixedArray MaterializedRingF productionProfile.commitmentWidth
abbrev MaterializedPublicInput :=
  FixedArray F
    (FullShape VerifierContext.candidateLogicalWidth
      VerifierContext.candidatePublicFits).publicWidth

def MaterializedRingF.ofRing (value : RingF) : MaterializedRingF :=
  FixedArray.ofFn value

def MaterializedRingF.toRing (value : MaterializedRingF) : RingF :=
  value.get

@[simp] theorem MaterializedRingF.toRing_ofRing (value : RingF) :
    (MaterializedRingF.ofRing value).toRing = value := by
  funext coefficient
  simp [MaterializedRingF.ofRing, MaterializedRingF.toRing]

def MaterializedRingK.ofRing (value : RingK) : MaterializedRingK :=
  FixedArray.ofFn value

def MaterializedRingK.toRing (value : MaterializedRingK) : RingK :=
  value.get

@[simp] theorem MaterializedRingK.toRing_ofRing (value : RingK) :
    (MaterializedRingK.ofRing value).toRing = value := by
  funext coefficient
  simp [MaterializedRingK.ofRing, MaterializedRingK.toRing]

def MaterializedCommitment.ofCommitment
    (value : PaperAlgebra.Commitment) : MaterializedCommitment :=
  FixedArray.ofFn fun row => MaterializedRingF.ofRing (value row)

def MaterializedCommitment.toCommitment
    (value : MaterializedCommitment) : PaperAlgebra.Commitment :=
  fun row => (value.get row).toRing

@[simp] theorem MaterializedCommitment.toCommitment_ofCommitment
    (value : PaperAlgebra.Commitment) :
    (MaterializedCommitment.ofCommitment value).toCommitment = value := by
  funext row coefficient
  simp [MaterializedCommitment.ofCommitment,
    MaterializedCommitment.toCommitment]

def MaterializedPublicInput.ofPublicInput
    (value : PublicInput
      (logicalWidth := VerifierContext.candidateLogicalWidth)
      (publicFits := VerifierContext.candidatePublicFits)) :
    MaterializedPublicInput :=
  FixedArray.ofFn value

def MaterializedPublicInput.toPublicInput
    (value : MaterializedPublicInput) :
    PublicInput (logicalWidth := VerifierContext.candidateLogicalWidth)
      (publicFits := VerifierContext.candidatePublicFits) :=
  value.get

@[simp] theorem MaterializedPublicInput.toPublicInput_ofPublicInput
    (value : PublicInput
      (logicalWidth := VerifierContext.candidateLogicalWidth)
      (publicFits := VerifierContext.candidatePublicFits)) :
    (MaterializedPublicInput.ofPublicInput value).toPublicInput = value := by
  funext column
  simp [MaterializedPublicInput.ofPublicInput,
    MaterializedPublicInput.toPublicInput]

def scan {Alpha : Type} (zero : Alpha)
    (step : Alpha → Fin SourceCount → Alpha) : List Alpha :=
  (List.finRange SourceCount).scanl step zero |>.tail

@[simp] theorem scan_length {Alpha : Type} (zero : Alpha)
    (step : Alpha → Fin SourceCount → Alpha) :
    (scan zero step).length = SourceCount := by
  simp [scan]

def prefixValue {Alpha : Type} (zero : Alpha)
    (step : Alpha → Fin SourceCount → Alpha) (count : Nat) : Alpha :=
  ((List.finRange SourceCount).take count).foldl step zero

/-- Every emitted index is the exact left-to-right fold through that source. -/
theorem scan_getElem? {Alpha : Type} (zero : Alpha)
    (step : Alpha → Fin SourceCount → Alpha) (index : Nat) :
    (scan zero step)[index]? =
      if index < SourceCount then
        some (prefixValue zero step (index + 1))
      else none := by
  unfold scan prefixValue
  rw [List.getElem?_tail, List.getElem?_scanl]
  simp only [List.length_finRange]
  by_cases bound : index < SourceCount
  · rw [if_pos (by omega), if_pos bound]
  · rw [if_neg (by omega), if_neg bound]

/-- The last emitted value is the exact full left-to-right fold. -/
theorem scan_getLast? {Alpha : Type} (zero : Alpha)
    (step : Alpha → Fin SourceCount → Alpha) :
    (scan zero step).getLast? =
      some ((List.finRange SourceCount).foldl step zero) := by
  unfold scan
  rw [List.getLast?_tail, if_neg]
  · exact List.getLast?_scanl
  · simp

private theorem scanl_map_hom
    {Alpha Beta : Type}
    (project : Alpha → Beta)
    (stepAlpha : Alpha → Fin SourceCount → Alpha)
    (stepBeta : Beta → Fin SourceCount → Beta)
    (sources : List (Fin SourceCount))
    (zeroAlpha : Alpha) (zeroBeta : Beta)
    (zeroEq : project zeroAlpha = zeroBeta)
    (stepEq : ∀ current source,
      project (stepAlpha current source) =
        stepBeta (project current) source) :
    (sources.scanl stepAlpha zeroAlpha).map project =
      sources.scanl stepBeta zeroBeta := by
  induction sources generalizing zeroAlpha zeroBeta with
  | nil => simp [zeroEq]
  | cons source rest inductionHypothesis =>
      simp only [List.scanl_cons, List.map_cons]
      rw [zeroEq]
      apply congrArg (List.cons zeroBeta)
      apply inductionHypothesis
      · calc
          project (stepAlpha zeroAlpha source) =
              stepBeta (project zeroAlpha) source := stepEq zeroAlpha source
          _ = stepBeta zeroBeta source := by rw [zeroEq]

theorem scan_map_hom
    {Alpha Beta : Type}
    (project : Alpha → Beta)
    (zeroAlpha : Alpha) (zeroBeta : Beta)
    (stepAlpha : Alpha → Fin SourceCount → Alpha)
    (stepBeta : Beta → Fin SourceCount → Beta)
    (zeroEq : project zeroAlpha = zeroBeta)
    (stepEq : ∀ current source,
      project (stepAlpha current source) =
        stepBeta (project current) source) :
    (scan zeroAlpha stepAlpha).map project = scan zeroBeta stepBeta := by
  unfold scan
  rw [List.map_tail,
    scanl_map_hom project stepAlpha stepBeta _ zeroAlpha zeroBeta zeroEq stepEq]

def commitmentStep (challenges : Fin SourceCount → RingF)
    (current : MaterializedCommitment) (source : Fin SourceCount) :
    MaterializedCommitment :=
  MaterializedCommitment.ofCommitment <|
    NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment.commitmentAdd
      current.toCommitment
      (NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment.commitmentAct
        (challenges source) (inputCommitment source))

def commitmentSemanticStep (challenges : Fin SourceCount → RingF)
    (current : PaperAlgebra.Commitment) (source : Fin SourceCount) :
    PaperAlgebra.Commitment :=
  NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment.commitmentAdd
    current
    (NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment.commitmentAct
      (challenges source) (inputCommitment source))

def commitmentPartials (challenges : Fin SourceCount → RingF) :
    List MaterializedCommitment :=
  scan
    (MaterializedCommitment.ofCommitment
      NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment.commitmentZero)
    (commitmentStep challenges)

def commitmentSemanticPartials (challenges : Fin SourceCount → RingF) :
    List PaperAlgebra.Commitment :=
  scan
    NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment.commitmentZero
    (commitmentSemanticStep challenges)

theorem commitmentPartials_semantics
    (challenges : Fin SourceCount → RingF) :
    (commitmentPartials challenges).map
        MaterializedCommitment.toCommitment =
      commitmentSemanticPartials challenges := by
  unfold commitmentPartials commitmentSemanticPartials
  apply scan_map_hom
  · simp
  · intro current source
    simp [commitmentStep, commitmentSemanticStep]

theorem commitmentPartials_indexed
    (challenges : Fin SourceCount → RingF) (index : Nat) :
    ((commitmentPartials challenges).map
      MaterializedCommitment.toCommitment)[index]? =
      if index < SourceCount then
        some (prefixValue
          NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment.commitmentZero
          (commitmentSemanticStep challenges) (index + 1))
      else none := by
  rw [commitmentPartials_semantics]
  simpa [commitmentSemanticPartials] using
    (scan_getElem?
      NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment.commitmentZero
      (commitmentSemanticStep challenges) index)

def publicInputStep (challenges : Fin SourceCount → RingF)
    (current : MaterializedPublicInput) (source : Fin SourceCount) :
    MaterializedPublicInput :=
  MaterializedPublicInput.ofPublicInput <|
    NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.PublicInput.publicAdd
      current.toPublicInput
      (NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.PublicInput.publicAct
        (challenges source) (inputPublicInput source))

def publicInputSemanticStep (challenges : Fin SourceCount → RingF)
    (current : PublicInput
      (logicalWidth := VerifierContext.candidateLogicalWidth)
      (publicFits := VerifierContext.candidatePublicFits))
    (source : Fin SourceCount) :
    PublicInput (logicalWidth := VerifierContext.candidateLogicalWidth)
      (publicFits := VerifierContext.candidatePublicFits) :=
  NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.PublicInput.publicAdd
    current
    (NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.PublicInput.publicAct
      (challenges source) (inputPublicInput source))

def publicInputPartials (challenges : Fin SourceCount → RingF) :
    List MaterializedPublicInput :=
  scan
    (MaterializedPublicInput.ofPublicInput
      NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.PublicInput.publicZero)
    (publicInputStep challenges)

def publicInputSemanticPartials (challenges : Fin SourceCount → RingF) :
    List (PublicInput (logicalWidth := VerifierContext.candidateLogicalWidth)
      (publicFits := VerifierContext.candidatePublicFits)) :=
  scan
    NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.PublicInput.publicZero
    (publicInputSemanticStep challenges)

theorem publicInputPartials_semantics
    (challenges : Fin SourceCount → RingF) :
    (publicInputPartials challenges).map
        MaterializedPublicInput.toPublicInput =
      publicInputSemanticPartials challenges := by
  unfold publicInputPartials publicInputSemanticPartials
  apply scan_map_hom
  · simp
  · intro current source
    simp [publicInputStep, publicInputSemanticStep]

theorem publicInputPartials_indexed
    (challenges : Fin SourceCount → RingF) (index : Nat) :
    ((publicInputPartials challenges).map
      MaterializedPublicInput.toPublicInput)[index]? =
      if index < SourceCount then
        some (prefixValue
          NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.PublicInput.publicZero
          (publicInputSemanticStep challenges) (index + 1))
      else none := by
  rw [publicInputPartials_semantics]
  simpa [publicInputSemanticPartials] using
    (scan_getElem?
      NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.PublicInput.publicZero
      (publicInputSemanticStep challenges) index)

def evaluationStep (challenges : Fin SourceCount → RingF)
    (values : Fin SourceCount → RingK)
    (current : MaterializedRingK) (source : Fin SourceCount) :
    MaterializedRingK :=
  MaterializedRingK.ofRing <|
    ringKAdd current.toRing
      (ringKMul (RingKAction.embedChallenge (challenges source))
        (values source))

def evaluationSemanticStep (challenges : Fin SourceCount → RingF)
    (values : Fin SourceCount → RingK)
    (current : RingK) (source : Fin SourceCount) : RingK :=
  ringKAdd current
    (ringKMul (RingKAction.embedChallenge (challenges source))
      (values source))

def evaluationPartials (challenges : Fin SourceCount → RingF)
    (values : Fin SourceCount → RingK) : List MaterializedRingK :=
  scan (MaterializedRingK.ofRing ringKZero)
    (evaluationStep challenges values)

def evaluationSemanticPartials (challenges : Fin SourceCount → RingF)
    (values : Fin SourceCount → RingK) : List RingK :=
  scan ringKZero (evaluationSemanticStep challenges values)

theorem evaluationPartials_semantics
    (challenges : Fin SourceCount → RingF)
    (values : Fin SourceCount → RingK) :
    (evaluationPartials challenges values).map MaterializedRingK.toRing =
      evaluationSemanticPartials challenges values := by
  unfold evaluationPartials evaluationSemanticPartials
  apply scan_map_hom
  · simp
  · intro current source
    simp [evaluationStep, evaluationSemanticStep]

theorem evaluationPartials_indexed
    (challenges : Fin SourceCount → RingF)
    (values : Fin SourceCount → RingK) (index : Nat) :
    ((evaluationPartials challenges values).map
      MaterializedRingK.toRing)[index]? =
      if index < SourceCount then
        some (prefixValue ringKZero
          (evaluationSemanticStep challenges values) (index + 1))
      else none := by
  rw [evaluationPartials_semantics]
  simpa [evaluationSemanticPartials] using
    (scan_getElem? ringKZero
      (evaluationSemanticStep challenges values) index)

private theorem combineCommitments_eq_foldr :
    ∀ {count : Nat} (challenges : Fin count → RingF)
      (values : Fin count → PaperAlgebra.Commitment),
    NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment.combineCommitments
        challenges values =
      (List.finRange count).foldr
        (fun source current =>
          NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment.commitmentAdd
            (NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment.commitmentAct
              (challenges source) (values source)) current)
        NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment.commitmentZero := by
  intro count
  induction count with
  | zero =>
      intro challenges values
      rfl
  | succ count inductionHypothesis =>
      intro challenges values
      simp only [
        NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment.combineCommitments,
        List.finRange_succ, List.foldr_cons, List.foldr_map]
      rw [inductionHypothesis
        (fun index => challenges index.succ)
        (fun index => values index.succ)]

private theorem combinePublicInputs_eq_foldr :
    ∀ {count : Nat} (challenges : Fin count → RingF)
      (values : Fin count → PublicInput
        (logicalWidth := VerifierContext.candidateLogicalWidth)
        (publicFits := VerifierContext.candidatePublicFits)),
    NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.PublicInput.combinePublicInputs
        challenges values =
      (List.finRange count).foldr
        (fun source current =>
          NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.PublicInput.publicAdd
            (NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.PublicInput.publicAct
              (challenges source) (values source)) current)
        NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.PublicInput.publicZero := by
  intro count
  induction count with
  | zero =>
      intro challenges values
      rfl
  | succ count inductionHypothesis =>
      intro challenges values
      simp only [
        NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.PublicInput.combinePublicInputs,
        List.finRange_succ, List.foldr_cons, List.foldr_map]
      rw [inductionHypothesis
        (fun index => challenges index.succ)
        (fun index => values index.succ)]

private theorem combineEvaluations_eq_foldr :
    ∀ {count : Nat} (challenges : Fin count → RingF)
      (values : Fin count → RingK),
    PiRLCFinite.combineEvaluation challenges values =
      (List.finRange count).foldr
        (fun source current =>
          ringKAdd
            (ringKMul (RingKAction.embedChallenge (challenges source))
              (values source)) current)
        ringKZero := by
  intro count
  induction count with
  | zero =>
      intro challenges values
      rfl
  | succ count inductionHypothesis =>
      intro challenges values
      simp only [PiRLCFinite.combineEvaluation, List.finRange_succ,
        List.foldr_cons, List.foldr_map]
      rw [inductionHypothesis
        (fun index => challenges index.succ)
        (fun index => values index.succ)]

private theorem commitmentAdd_assoc
    (left middle right : PaperAlgebra.Commitment) :
    NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment.commitmentAdd
        (NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment.commitmentAdd
          left middle) right =
      NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment.commitmentAdd
        left
        (NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment.commitmentAdd
          middle right) := by
  funext row lane
  exact ConcreteCarrier.baseLaws.add_assoc _ _ _

private theorem commitmentZero_add (value : PaperAlgebra.Commitment) :
    NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment.commitmentAdd
        NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment.commitmentZero
        value = value := by
  funext row lane
  exact ConcreteCarrier.baseLaws.zero_add _

private theorem commitmentAdd_zero (value : PaperAlgebra.Commitment) :
    NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment.commitmentAdd
        value
        NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment.commitmentZero =
      value := by
  funext row lane
  exact ConcreteCarrier.baseLaws.add_zero _

private theorem publicAdd_assoc
    (left middle right : PublicInput
      (logicalWidth := VerifierContext.candidateLogicalWidth)
      (publicFits := VerifierContext.candidatePublicFits)) :
    NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.PublicInput.publicAdd
        (NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.PublicInput.publicAdd
          left middle) right =
      NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.PublicInput.publicAdd
        left
        (NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.PublicInput.publicAdd
          middle right) := by
  funext column
  exact ConcreteCarrier.baseLaws.add_assoc _ _ _

private theorem publicZero_add
    (value : PublicInput
      (logicalWidth := VerifierContext.candidateLogicalWidth)
      (publicFits := VerifierContext.candidatePublicFits)) :
    NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.PublicInput.publicAdd
        NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.PublicInput.publicZero
        value = value := by
  funext column
  exact ConcreteCarrier.baseLaws.zero_add _

private theorem publicAdd_zero
    (value : PublicInput
      (logicalWidth := VerifierContext.candidateLogicalWidth)
      (publicFits := VerifierContext.candidatePublicFits)) :
    NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.PublicInput.publicAdd
        value
        NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.PublicInput.publicZero =
      value := by
  funext column
  exact ConcreteCarrier.baseLaws.add_zero _

private theorem ringKAdd_assoc (left middle right : RingK) :
    ringKAdd (ringKAdd left middle) right =
      ringKAdd left (ringKAdd middle right) := by
  funext lane
  exact ConcreteCarrier.extensionLaws.add_assoc _ _ _

private theorem ringKZero_add (value : RingK) :
    ringKAdd ringKZero value = value := by
  funext lane
  exact ConcreteCarrier.extensionLaws.zero_add _

private theorem ringKAdd_zero (value : RingK) :
    ringKAdd value ringKZero = value := by
  funext lane
  exact ConcreteCarrier.extensionLaws.add_zero _

local instance : Std.Associative
    (NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment.commitmentAdd :
      PaperAlgebra.Commitment → PaperAlgebra.Commitment →
        PaperAlgebra.Commitment) :=
  ⟨commitmentAdd_assoc⟩

local instance : Std.LawfulIdentity
    (NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment.commitmentAdd :
      PaperAlgebra.Commitment → PaperAlgebra.Commitment →
        PaperAlgebra.Commitment)
    (NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment.commitmentZero :
      PaperAlgebra.Commitment) where
  left_id := commitmentZero_add
  right_id := commitmentAdd_zero

local instance : Std.Associative
    (NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.PublicInput.publicAdd :
      PublicInput (logicalWidth := VerifierContext.candidateLogicalWidth)
          (publicFits := VerifierContext.candidatePublicFits) →
        PublicInput (logicalWidth := VerifierContext.candidateLogicalWidth)
          (publicFits := VerifierContext.candidatePublicFits) →
        PublicInput (logicalWidth := VerifierContext.candidateLogicalWidth)
          (publicFits := VerifierContext.candidatePublicFits)) :=
  ⟨publicAdd_assoc⟩

local instance : Std.LawfulIdentity
    (NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.PublicInput.publicAdd :
      PublicInput (logicalWidth := VerifierContext.candidateLogicalWidth)
          (publicFits := VerifierContext.candidatePublicFits) →
        PublicInput (logicalWidth := VerifierContext.candidateLogicalWidth)
          (publicFits := VerifierContext.candidatePublicFits) →
        PublicInput (logicalWidth := VerifierContext.candidateLogicalWidth)
          (publicFits := VerifierContext.candidatePublicFits))
    (NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.PublicInput.publicZero :
      PublicInput (logicalWidth := VerifierContext.candidateLogicalWidth)
        (publicFits := VerifierContext.candidatePublicFits)) where
  left_id := publicZero_add
  right_id := publicAdd_zero

local instance : Std.Associative ringKAdd := ⟨ringKAdd_assoc⟩

local instance : Std.LawfulIdentity ringKAdd ringKZero where
  left_id := ringKZero_add
  right_id := ringKAdd_zero

private theorem commitmentFoldl_eq_combined
    (challenges : Fin SourceCount → RingF) :
    (List.finRange SourceCount).foldl
        (commitmentSemanticStep challenges)
        NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment.commitmentZero =
      combinedCommitment challenges := by
  unfold commitmentSemanticStep
  rw [← List.foldl_map, ← List.foldr_eq_foldl, List.foldr_map]
  exact (combineCommitments_eq_foldr challenges inputCommitment).symm

private theorem publicInputFoldl_eq_combined
    (challenges : Fin SourceCount → RingF) :
    (List.finRange SourceCount).foldl
        (publicInputSemanticStep challenges)
        NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.PublicInput.publicZero =
      combinedPublicInput challenges := by
  unfold publicInputSemanticStep
  rw [← List.foldl_map, ← List.foldr_eq_foldl, List.foldr_map]
  exact (combinePublicInputs_eq_foldr challenges inputPublicInput).symm

private theorem evaluationFoldl_eq_combined
    (challenges : Fin SourceCount → RingF)
    (values : Fin SourceCount → RingK) :
    (List.finRange SourceCount).foldl
        (evaluationSemanticStep challenges values) ringKZero =
      PiRLCFinite.combineEvaluation challenges values := by
  unfold evaluationSemanticStep
  rw [← List.foldl_map, ← List.foldr_eq_foldl, List.foldr_map]
  exact (combineEvaluations_eq_foldr challenges values).symm

theorem commitmentPartials_getLast?
    (challenges : Fin SourceCount → RingF) :
    ((commitmentPartials challenges).map
      MaterializedCommitment.toCommitment).getLast? =
      some (combinedCommitment challenges) := by
  rw [commitmentPartials_semantics]
  unfold commitmentSemanticPartials
  rw [scan_getLast?, commitmentFoldl_eq_combined]

theorem publicInputPartials_getLast?
    (challenges : Fin SourceCount → RingF) :
    ((publicInputPartials challenges).map
      MaterializedPublicInput.toPublicInput).getLast? =
      some (combinedPublicInput challenges) := by
  rw [publicInputPartials_semantics]
  unfold publicInputSemanticPartials
  rw [scan_getLast?, publicInputFoldl_eq_combined]

theorem evaluationPartials_getLast?
    (challenges : Fin SourceCount → RingF)
    (values : Fin SourceCount → RingK) :
    ((evaluationPartials challenges values).map
      MaterializedRingK.toRing).getLast? =
      some (PiRLCFinite.combineEvaluation challenges values) := by
  rw [evaluationPartials_semantics]
  unfold evaluationSemanticPartials
  rw [scan_getLast?, evaluationFoldl_eq_combined]

def evalKPartials (challenges : Fin SourceCount → RingF) :
    List MaterializedRingK :=
  evaluationPartials challenges fun source => (inputEvaluation source).pad

def evalAPartials (challenges : Fin SourceCount → RingF)
    (matrix : Fin productionShape.matrixCount) : List MaterializedRingK :=
  evaluationPartials challenges fun source =>
    (inputEvaluation source).matrix matrix

theorem evalKPartials_getLast? (challenges : Fin SourceCount → RingF) :
    ((evalKPartials challenges).map MaterializedRingK.toRing).getLast? =
      some (combinedEvaluation challenges).pad := by
  unfold evalKPartials combinedEvaluation
  exact evaluationPartials_getLast? challenges fun source =>
    (inputEvaluation source).pad

theorem evalAPartials_getLast? (challenges : Fin SourceCount → RingF)
    (matrix : Fin productionShape.matrixCount) :
    ((evalAPartials challenges matrix).map
      MaterializedRingK.toRing).getLast? =
      some ((combinedEvaluation challenges).matrix matrix) := by
  unfold evalAPartials combinedEvaluation
  exact evaluationPartials_getLast? challenges fun source =>
    (inputEvaluation source).matrix matrix

end NightstreamFPrime.Export.Stage1.PiRLCPartialTrace

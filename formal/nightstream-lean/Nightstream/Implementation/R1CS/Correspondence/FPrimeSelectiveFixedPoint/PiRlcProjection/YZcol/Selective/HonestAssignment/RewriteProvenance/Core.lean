import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.DerivedProgram.Core

/-!
Executable provenance checks for the source and derived columns read by the
bounded selective rewrite program.

Owns: source-closure membership and derived-slot coverage for a supplied list
of decoded rewrite steps, plus the Boolean-check-to-Proposition bridge.

Does not own: the artifact partition checks, honest source semantics,
assignment execution, selected-row satisfaction, or projection authority.

Emits constraints: no.

| Provenance leaf | Mathematical obligation | Authority class |
|---|---|---|
| source closure | every source term belongs to the compiler closure | checked |
| slot coverage | every derived input and output belongs to the slot registry | checked |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge

def compilerClosure : List Nat :=
  Nightstream.Implementation.R1CS.Program.knownAfter
    SourceDecode.compilerKnownColumns SourceDecode.compilerDefinitions

def LinearKnown
    (linear : SourceDecode.DecodedSourceLinearCombination) : Prop :=
  ∀ term ∈ linear.programTerms, term.1 ∈ compilerClosure

def FactorKnown (factor : DecodedProductFactor) : Prop :=
  LinearKnown factor.left ∧ LinearKnown factor.right

def StepSourcesKnown (step : DecodedRewriteStep) : Prop :=
  LinearKnown step.base ∧
    (∀ factor ∈ step.factors, FactorKnown factor) ∧
    match step.output with
    | .source linear => LinearKnown linear
    | .derivedProductSum _ => True

def StepSlotsCovered (step : DecodedRewriteStep) : Prop :=
  (match step.output with
    | .source _ => True
    | .derivedProductSum slot => slot ∈ decodedDerivedSlots) ∧
  (match step.previous with
    | none => True
    | some slot => slot ∈ decodedDerivedSlots)

/-! ## Proof-free certificate representation -/

def linearColumns
    (linear : SourceDecode.DecodedSourceLinearCombination) : List Nat :=
  linear.programTerms.map Prod.fst

def factorColumnShape
    (factor : DecodedProductFactor) : List Nat × List Nat :=
  (linearColumns factor.left, linearColumns factor.right)

structure DerivedSlotShape where
  compilerIndex : Nat
  start : Nat
  width : Nat
deriving DecidableEq, Repr

def derivedSlotShape (slot : DecodedDerivedSlot) : DerivedSlotShape :=
  { compilerIndex := slot.compilerIndex
    start := slot.start
    width := slot.width }

inductive ProvenanceOutputShape where
  | source (columns : List Nat)
  | derived (slot : DerivedSlotShape)
deriving DecidableEq, Repr

structure RewriteProvenanceShape where
  baseColumns : List Nat
  factorColumns : List (List Nat × List Nat)
  output : ProvenanceOutputShape
  previous : Option DerivedSlotShape
deriving DecidableEq, Repr

/-- Compact data consumed by native artifact checking. All proof fields,
coefficients, source-row blocks, and emitted-row metadata are erased. -/
def rewriteProvenanceShape
    (step : DecodedRewriteStep) : RewriteProvenanceShape :=
  { baseColumns := linearColumns step.base
    factorColumns := step.factors.map factorColumnShape
    output :=
      match step.output with
      | .source linear => .source (linearColumns linear)
      | .derivedProductSum slot => .derived (derivedSlotShape slot)
    previous := step.previous.map derivedSlotShape }

def compilerClosureIndex : Std.HashMap Nat Unit :=
  Std.HashMap.unitOfList compilerClosure

def compilerColumnKnownCheck (column : Nat) : Bool :=
  compilerClosureIndex.contains column

private theorem compilerColumnKnown_of_check_true
    {column : Nat} (checked : compilerColumnKnownCheck column = true) :
    column ∈ compilerClosure := by
  apply List.mem_of_elem_eq_true
  simpa only [compilerColumnKnownCheck, compilerClosureIndex,
    Std.HashMap.contains_unitOfList] using checked

def columnsKnownCheck (columns : List Nat) : Bool :=
  columns.all compilerColumnKnownCheck

private theorem columnsKnown_of_check_true
    {columns : List Nat} (checked : columnsKnownCheck columns = true) :
    ∀ column ∈ columns, column ∈ compilerClosure := by
  intro column member
  apply compilerColumnKnown_of_check_true
  have allChecked : columns.all compilerColumnKnownCheck = true := by
    simpa only [columnsKnownCheck] using checked
  exact (List.all_eq_true.mp allChecked) column member

theorem linearKnown_of_shape_check_true
    {linear : SourceDecode.DecodedSourceLinearCombination}
    (checked : columnsKnownCheck (linearColumns linear) = true) :
    LinearKnown linear := by
  intro term member
  apply columnsKnown_of_check_true checked term.1
  exact List.mem_map.mpr ⟨term, member, rfl⟩

def factorColumnsKnownCheck
    (shape : List Nat × List Nat) : Bool :=
  columnsKnownCheck shape.1 && columnsKnownCheck shape.2

private theorem factorKnown_of_shape_check_true
    {factor : DecodedProductFactor}
    (checked : factorColumnsKnownCheck (factorColumnShape factor) = true) :
    FactorKnown factor := by
  have parts :
      columnsKnownCheck (linearColumns factor.left) = true ∧
        columnsKnownCheck (linearColumns factor.right) = true := by
    simpa only [factorColumnsKnownCheck, factorColumnShape,
      Bool.and_eq_true] using checked
  exact ⟨linearKnown_of_shape_check_true parts.1,
    linearKnown_of_shape_check_true parts.2⟩

private theorem factorsKnown_of_shape_check_true
    {factors : List DecodedProductFactor}
    (checked :
      (factors.map factorColumnShape).all factorColumnsKnownCheck = true) :
    ∀ factor ∈ factors, FactorKnown factor := by
  intro factor member
  apply factorKnown_of_shape_check_true
  apply (List.all_eq_true.mp checked) (factorColumnShape factor)
  exact List.mem_map.mpr ⟨factor, member, rfl⟩

def derivedSlotShapes : List DerivedSlotShape :=
  decodedDerivedSlots.map derivedSlotShape

def derivedSlotShapeKey (shape : DerivedSlotShape) : Nat × Nat × Nat :=
  (shape.compilerIndex, shape.start, shape.width)

def derivedSlotShapeKeys : List (Nat × Nat × Nat) :=
  derivedSlotShapes.map derivedSlotShapeKey

def derivedSlotShapeIndex : Std.HashMap (Nat × Nat × Nat) Unit :=
  Std.HashMap.unitOfList derivedSlotShapeKeys

def derivedSlotShapeMemberCheck (shape : DerivedSlotShape) : Bool :=
  derivedSlotShapeIndex.contains (derivedSlotShapeKey shape)

private theorem derivedSlotShape_eq_of_key_eq
    {left right : DerivedSlotShape}
    (equal : derivedSlotShapeKey left = derivedSlotShapeKey right) :
    left = right := by
  cases left
  cases right
  simp_all [derivedSlotShapeKey]

private theorem derivedSlot_eq_of_shape_eq
    {left right : DecodedDerivedSlot}
    (equal : derivedSlotShape left = derivedSlotShape right) :
    left = right := by
  cases left
  cases right
  simp_all [derivedSlotShape]

private theorem derivedSlotMember_of_shape_check_true
    {slot : DecodedDerivedSlot}
    (checked : derivedSlotShapeMemberCheck (derivedSlotShape slot) = true) :
    slot ∈ decodedDerivedSlots := by
  have keyMember :
      derivedSlotShapeKey (derivedSlotShape slot) ∈ derivedSlotShapeKeys := by
    apply List.mem_of_elem_eq_true
    simpa only [derivedSlotShapeMemberCheck, derivedSlotShapeIndex,
      Std.HashMap.contains_unitOfList] using checked
  rcases List.mem_map.mp keyMember with
    ⟨candidateShape, candidateShapeMember, keyEqual⟩
  have shapeEqual : candidateShape = derivedSlotShape slot :=
    derivedSlotShape_eq_of_key_eq keyEqual
  have shapeMember : derivedSlotShape slot ∈ derivedSlotShapes := by
    simpa [shapeEqual] using candidateShapeMember
  rcases List.mem_map.mp shapeMember with
    ⟨candidate, candidateMember, decodedShapeEqual⟩
  have equal : candidate = slot :=
    derivedSlot_eq_of_shape_eq decodedShapeEqual
  simpa [equal] using candidateMember

def provenanceSourcesCheck (shape : RewriteProvenanceShape) : Bool :=
  columnsKnownCheck shape.baseColumns &&
    (shape.factorColumns.all factorColumnsKnownCheck &&
      match shape.output with
      | .source columns => columnsKnownCheck columns
      | .derived _ => true)

def provenanceSlotsCheck (shape : RewriteProvenanceShape) : Bool :=
  (match shape.output with
    | .source _ => true
    | .derived slot => derivedSlotShapeMemberCheck slot) &&
  (match shape.previous with
    | none => true
    | some slot => derivedSlotShapeMemberCheck slot)

private theorem stepSourcesKnown_of_shape_check_true
    {step : DecodedRewriteStep}
    (checked : provenanceSourcesCheck (rewriteProvenanceShape step) = true) :
    StepSourcesKnown step := by
  cases outputEq : step.output with
  | source linear =>
      have parts :
          columnsKnownCheck (linearColumns step.base) = true ∧
            ((step.factors.map factorColumnShape).all
                factorColumnsKnownCheck = true ∧
              columnsKnownCheck (linearColumns linear) = true) := by
        simpa only [provenanceSourcesCheck, rewriteProvenanceShape,
          outputEq, Bool.and_eq_true] using checked
      unfold StepSourcesKnown
      rw [outputEq]
      exact ⟨linearKnown_of_shape_check_true parts.1,
        factorsKnown_of_shape_check_true parts.2.1,
        linearKnown_of_shape_check_true parts.2.2⟩
  | derivedProductSum slot =>
      have parts :
          columnsKnownCheck (linearColumns step.base) = true ∧
            (step.factors.map factorColumnShape).all
              factorColumnsKnownCheck = true := by
        simpa only [provenanceSourcesCheck, rewriteProvenanceShape,
          outputEq, Bool.and_eq_true, and_true] using checked
      unfold StepSourcesKnown
      rw [outputEq]
      exact ⟨linearKnown_of_shape_check_true parts.1,
        factorsKnown_of_shape_check_true parts.2, trivial⟩

private theorem stepSlotsCovered_of_shape_check_true
    {step : DecodedRewriteStep}
    (checked : provenanceSlotsCheck (rewriteProvenanceShape step) = true) :
    StepSlotsCovered step := by
  rw [provenanceSlotsCheck] at checked
  have checks :
      (match (rewriteProvenanceShape step).output with
        | .source _ => true
        | .derived slot => derivedSlotShapeMemberCheck slot) = true ∧
      (match (rewriteProvenanceShape step).previous with
        | none => true
        | some slot => derivedSlotShapeMemberCheck slot) = true := by
    simpa only [Bool.and_eq_true] using checked
  unfold StepSlotsCovered
  cases outputEq : step.output with
  | source linear =>
      cases previousEq : step.previous with
      | none => exact ⟨trivial, trivial⟩
      | some previous =>
          have previousCheck :
              derivedSlotShapeMemberCheck (derivedSlotShape previous) = true := by
            simpa only [rewriteProvenanceShape, outputEq, previousEq] using
              checks.2
          exact ⟨trivial,
            derivedSlotMember_of_shape_check_true previousCheck⟩
  | derivedProductSum output =>
      cases previousEq : step.previous with
      | none =>
          have outputCheck :
              derivedSlotShapeMemberCheck (derivedSlotShape output) = true := by
            simpa only [rewriteProvenanceShape, outputEq] using checks.1
          exact ⟨derivedSlotMember_of_shape_check_true outputCheck, trivial⟩
      | some previous =>
          have outputCheck :
              derivedSlotShapeMemberCheck (derivedSlotShape output) = true := by
            simpa only [rewriteProvenanceShape, outputEq] using checks.1
          have previousCheck :
              derivedSlotShapeMemberCheck (derivedSlotShape previous) = true := by
            simpa only [rewriteProvenanceShape, previousEq] using checks.2
          exact ⟨derivedSlotMember_of_shape_check_true outputCheck,
            derivedSlotMember_of_shape_check_true previousCheck⟩

def rewriteProvenanceShapeCheck
    (shapes : List RewriteProvenanceShape) : Bool :=
  shapes.all fun shape =>
    provenanceSourcesCheck shape && provenanceSlotsCheck shape

/-- Generic kernel bridge from a compact provenance certificate to the
original decoded rewrite records. -/
theorem rewriteProvenance_of_shape_check_true
    {steps : List DecodedRewriteStep}
    (checked :
      rewriteProvenanceShapeCheck (steps.map rewriteProvenanceShape) = true) :
    ∀ step ∈ steps, StepSourcesKnown step ∧ StepSlotsCovered step := by
  intro step member
  have shapeMember :
      rewriteProvenanceShape step ∈ steps.map rewriteProvenanceShape :=
    List.mem_map.mpr ⟨step, member, rfl⟩
  have allChecked :
      (steps.map rewriteProvenanceShape).all (fun shape =>
        provenanceSourcesCheck shape && provenanceSlotsCheck shape) = true := by
    simpa only [rewriteProvenanceShapeCheck] using checked
  have stepChecked :=
    (List.all_eq_true.mp allChecked) (rewriteProvenanceShape step) shapeMember
  have parts :
      provenanceSourcesCheck (rewriteProvenanceShape step) = true ∧
        provenanceSlotsCheck (rewriteProvenanceShape step) = true := by
    simpa only [Bool.and_eq_true] using stepChecked
  exact ⟨stepSourcesKnown_of_shape_check_true parts.1,
    stepSlotsCovered_of_shape_check_true parts.2⟩

def rewriteProvenanceChunk0Data : List RewriteProvenanceShape :=
  derivedProgramChunk0.map rewriteProvenanceShape

def rewriteProvenanceChunk1Data : List RewriteProvenanceShape :=
  derivedProgramChunk1.map rewriteProvenanceShape

def rewriteProvenanceChunk2Data : List RewriteProvenanceShape :=
  derivedProgramChunk2.map rewriteProvenanceShape

def rewriteProvenanceChunk3Data : List RewriteProvenanceShape :=
  derivedProgramChunk3.map rewriteProvenanceShape

def rewriteProvenanceChunk4Data : List RewriteProvenanceShape :=
  derivedProgramChunk4.map rewriteProvenanceShape

theorem rewriteProvenanceChunkLengthsExact :
    rewriteProvenanceChunk0Data.length = 250 ∧
      rewriteProvenanceChunk1Data.length = 250 ∧
      rewriteProvenanceChunk2Data.length = 250 ∧
      rewriteProvenanceChunk3Data.length = 250 ∧
      rewriteProvenanceChunk4Data.length = 250 := by
  simpa only [rewriteProvenanceChunk0Data, rewriteProvenanceChunk1Data,
    rewriteProvenanceChunk2Data, rewriteProvenanceChunk3Data,
    rewriteProvenanceChunk4Data, List.length_map] using
    derivedProgramChunkLengthsExact

theorem rewriteProvenanceChunksExact :
    rewriteProvenanceChunk0Data ++
        (rewriteProvenanceChunk1Data ++
          (rewriteProvenanceChunk2Data ++
            (rewriteProvenanceChunk3Data ++ rewriteProvenanceChunk4Data))) =
      decodedRewriteSteps.map rewriteProvenanceShape := by
  simpa only [rewriteProvenanceChunk0Data, rewriteProvenanceChunk1Data,
    rewriteProvenanceChunk2Data, rewriteProvenanceChunk3Data,
    rewriteProvenanceChunk4Data, List.map_append] using
    congrArg (List.map rewriteProvenanceShape) derivedProgramChunksExact

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment

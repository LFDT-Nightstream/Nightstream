import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcHandoff
import Nightstream.Implementation.R1CS.Canonical.KPiRlcProjectionTranscript

/-!
Contract: connect the selected fixed-active `Pi_RLC` sampler outputs to the
public quotient-identity occurrence.

Owns:
- the exact fifteen-by-fifty-four challenge-column projection from the
  canonical selector;
- a fresh physical duplex receipt after the sampler allocation that preserves
  the sampler's final lanes and cursor;
- replacement of every free public quotient challenge list by those physical
  sampler outputs; and
- the combined sampler-plus-quotient row program and its semantic handoff.

Does not own the authoritative PiCCS input/output carrier projection, quotient
witness construction, the delayed old-point bridge, a probability bound, or
the complete `nifsVerify` recipe.

Assurance tier: model-level canonical encoding.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcQuotientHandoff

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding

abbrev Params := productionGlobalParams
abbrev Arity := Nightstream.SuperNeo.Folding.Nifs.PaperProfile.arity

/-! ## Exact sampler challenge columns -/

/-- The typed paper source index and the sampler coordinate are the same
fifteen-element index. -/
def samplerCoordinate (index : Fin Arity.total) :
    Fin PiRlcCanonicalSamplerProgram.coordinateCount :=
  ⟨index.val, by
    simpa [PiRlcCanonicalSamplerProgram.coordinateCount] using index.isLt⟩

/-- Every coefficient of one public `Pi_RLC` challenge is the corresponding
centered selector output column. -/
def challengeColumns (duplexBase : Nat) (index : Fin Arity.total) :
    List Nat :=
  List.ofFn fun position : Fin PiRlcCanonicalSelector.outputCount =>
    PiRlcCanonicalSelector.outputColumn
      (PiRlcCanonicalSamplerProgram.selectorBase duplexBase)
      (samplerCoordinate index) position

@[simp] theorem challengeColumns_length
    (duplexBase : Nat) (index : Fin Arity.total) :
    (challengeColumns duplexBase index).length = Concrete.ringDegree := by
  simp [challengeColumns, PiRlcCanonicalSelector.outputCount,
    Concrete.ringDegree]

theorem challengeColumns_values
    (duplexBase : Nat) (index : Fin Arity.total)
    (assignment : Nat → Nat) :
    (challengeColumns duplexBase index).map assignment =
      PiRlcCanonicalSamplerSound.physicalOutputValues
        (PiRlcCanonicalSamplerProgram.selectorBase duplexBase)
        (samplerCoordinate index) assignment := by
  rfl

/-! ## Fresh post-sampler duplex receipt -/

/-- First column after the complete selector allocation. -/
def quotientTranscriptBase (duplexBase : Nat) : Nat :=
  PiRlcCanonicalSamplerProgram.selectorBase duplexBase +
    PiRlcCanonicalSamplerProgram.coordinateCount *
      PiRlcCanonicalSelector.scalarAuxiliaryCount

/-- The sampler's final state and cursor, with a fresh entry list whose
physical permutations begin after every sampler-owned column. -/
def quotientPrior (duplexBase : Nat) (lanes : State) :
    SymbolicDuplex.Builder :=
  SymbolicDuplex.start
    (PiRlcCanonicalSymbolicMachineHonest.fixedBuilder
      duplexBase lanes).lanes
    (PiRlcCanonicalSymbolicMachineHonest.fixedBuilder
      duplexBase lanes).absorbed

@[simp] theorem quotientPrior_entries
    (duplexBase : Nat) (lanes : State) :
    (quotientPrior duplexBase lanes).entries = [] := by
  rfl

@[simp] theorem quotientPrior_absorbed
    (duplexBase : Nat) (lanes : State) :
    (quotientPrior duplexBase lanes).absorbed = 0 := by
  rfl

theorem decoded_quotientPrior_eq_fixedBuilder
    (duplexBase : Nat) (lanes : State) (assignment : Nat → Nat) :
    SymbolicDuplexSemantics.decodedBuilder assignment
        (quotientPrior duplexBase lanes) =
      SymbolicDuplexSemantics.decodedBuilder assignment
        (PiRlcCanonicalSymbolicMachineHonest.fixedBuilder duplexBase lanes) := by
  have decodedStart
      (builder : SymbolicDuplex.Builder) :
      SymbolicDuplexSemantics.decodedBuilder assignment
          (SymbolicDuplex.start builder.lanes builder.absorbed) =
        SymbolicDuplexSemantics.decodedBuilder assignment builder := by
    rfl
  exact decodedStart
    (PiRlcCanonicalSymbolicMachineHonest.fixedBuilder duplexBase lanes)

theorem challengeColumn_mem_selectorAllocation
    (duplexBase : Nat) (index : Fin Arity.total)
    (position : Fin PiRlcCanonicalSelector.outputCount) :
    PiRlcCanonicalSelector.outputColumn
        (PiRlcCanonicalSamplerProgram.selectorBase duplexBase)
        (samplerCoordinate index) position ∈
      PiRlcCanonicalSelector.allocation
        (PiRlcCanonicalSamplerProgram.selectorBase duplexBase)
        PiRlcCanonicalSamplerProgram.coordinateCount := by
  rw [PiRlcCanonicalSelector.allocation_mem_iff]
  unfold PiRlcCanonicalSelector.outputColumn
    PiRlcCanonicalSelector.positionBase
    PiRlcCanonicalSelector.scalarBase
  have coordinateLt := (samplerCoordinate index).isLt
  have positionLt := position.isLt
  simp only [PiRlcCanonicalSelector.outputCount,
    PiRlcCanonicalSelector.positionAuxiliaryCount,
    PiRlcCanonicalSelector.scalarAuxiliaryCount] at coordinateLt positionLt ⊢
  omega

theorem challengeColumn_below_quotientTranscriptBase
    (duplexBase : Nat) (index : Fin Arity.total) (column : Nat)
    (member : column ∈ challengeColumns duplexBase index) :
    column < quotientTranscriptBase duplexBase := by
  unfold challengeColumns at member
  rcases List.mem_ofFn.mp member with ⟨position, rfl⟩
  have allocated :=
    challengeColumn_mem_selectorAllocation duplexBase index position
  rw [PiRlcCanonicalSelector.allocation_mem_iff] at allocated
  simpa [quotientTranscriptBase] using allocated.2

/-! ## Sampler-bound quotient source -/

/-- Replace the quotient occurrence's free challenge lists by the canonical
sampler outputs.  All other fields remain owned by their call-frame
projection. -/
def samplerBoundSource
    (duplexBase : Nat)
    (source : KPiRlcSemanticBinding.SourceColumns Params Arity 13) :
    KPiRlcSemanticBinding.SourceColumns Params Arity 13 :=
  { source with challenges := challengeColumns duplexBase }

@[simp] theorem samplerBoundSource_challenges
    (duplexBase : Nat)
    (source : KPiRlcSemanticBinding.SourceColumns Params Arity 13)
    (index : Fin Arity.total) :
    (samplerBoundSource duplexBase source).challenges index =
      challengeColumns duplexBase index := by
  rfl

theorem samplerBoundSource_valid
    (duplexBase : Nat)
    (source : KPiRlcSemanticBinding.SourceColumns Params Arity 13)
    (valid : source.Valid) :
    (samplerBoundSource duplexBase source).Valid := by
  refine
    { challengeWidth := challengeColumns_length duplexBase
      inputWidth := ?_
      outputWidth := ?_ }
  · exact valid.inputWidth
  · exact valid.outputWidth

/-- Bounds for the call-frame-owned coefficient and quotient columns.  The
challenge bound is intentionally absent: it is derived from the sampler
allocation above. -/
structure PayloadColumnsBelow
    (base : Nat)
    (source : KPiRlcSemanticBinding.SourceColumns Params Arity 13)
    (quotients : PublicRole 13 → List Nat) : Prop where
  sourceInput :
    ∀ index role column,
      column ∈ (source.inputs index).at role → column < base
  output :
    ∀ role column,
      column ∈ source.output.at role → column < base
  quotient :
    ∀ role column,
      column ∈ quotients role → column < base

/-- The quotient transcript starts after the full sampler allocation and
continues from the sampler's final state without reusing its entry indices. -/
def projectionInput
    (duplexBase : Nat) (lanes : State)
    (source : KPiRlcSemanticBinding.SourceColumns Params Arity 13)
    (quotients : PublicRole 13 → List Nat) :
    KPiRlcProjectionTranscript.Input Params Arity 13 where
  transcriptBase := quotientTranscriptBase duplexBase
  prior := quotientPrior duplexBase lanes
  source := samplerBoundSource duplexBase source
  quotients := quotients

theorem projectionInput_columnsBelow
    (duplexBase : Nat) (lanes : State)
    (source : KPiRlcSemanticBinding.SourceColumns Params Arity 13)
    (quotients : PublicRole 13 → List Nat)
    (below :
      PayloadColumnsBelow (quotientTranscriptBase duplexBase)
        source quotients) :
    (projectionInput duplexBase lanes source quotients).ColumnsBelowTranscript := by
  refine
    { challenge := ?_
      sourceInput := ?_
      output := ?_
      quotient := ?_ }
  · intro index column member
    exact challengeColumn_below_quotientTranscriptBase
      duplexBase index column member
  · exact below.sourceInput
  · exact below.output
  · exact below.quotient

theorem projectionInput_valid
    (duplexBase : Nat) (lanes : State)
    (source : KPiRlcSemanticBinding.SourceColumns Params Arity 13)
    (quotients : PublicRole 13 → List Nat)
    (sourceValid : source.Valid)
    (quotientWidth : ∀ role, (quotients role).length = 53) :
    (KPiRlcProjectionTranscript.projectionColumns
      (projectionInput duplexBase lanes source quotients)).Valid := by
  exact
    KPiRlcProjectionTranscript.projectionColumns_valid
      (projectionInput duplexBase lanes source quotients)
      (samplerBoundSource_valid duplexBase source sourceValid)
      quotientWidth

/-! ## Selected PiCCS/sampler/quotient composition -/

/-- The selected quotient suffix starts from the final physical sampler lanes,
not from a digest or caller-supplied transcript state. -/
def selectedProjectionInput
    {rowsCount columns degree : Nat}
    (profile : PaperNifsPiRlcHandoff.SelectiveProfile rowsCount columns)
    (duplexBase : Nat)
    (piCcsInput :
      KPiCcsTranscript.Input
        (PaperNifsPiRlcHandoff.SelectedShape profile) degree)
    (source : KPiRlcSemanticBinding.SourceColumns Params Arity 13)
    (quotients : PublicRole 13 → List Nat) :
    KPiRlcProjectionTranscript.Input Params Arity 13 :=
  projectionInput duplexBase
    (KPiCcsTranscript.replay piCcsInput).afterOutput.lanes
    source quotients

/-- Exact physical composition through the public quotient occurrence. -/
def rows
    {rowsCount columns degree : Nat}
    (profile : PaperNifsPiRlcHandoff.SelectiveProfile rowsCount columns)
    (duplexBase : Nat) (constants : Poseidon2Schedule.Constants)
    (piCcsInput :
      KPiCcsTranscript.Input
        (PaperNifsPiRlcHandoff.SelectedShape profile) degree)
    (source : KPiRlcSemanticBinding.SourceColumns Params Arity 13)
    (quotients : PublicRole 13 → List Nat)
    (sourceValid : source.Valid)
    (quotientWidth : ∀ role, (quotients role).length = 53) :
    List Row :=
  PaperNifsPiRlcHandoff.rows profile duplexBase constants piCcsInput ++
    KPiRlcProjectionTranscript.rows constants
      (selectedProjectionInput profile duplexBase piCcsInput source quotients)
      (projectionInput_valid duplexBase
        (KPiCcsTranscript.replay piCcsInput).afterOutput.lanes
        source quotients sourceValid quotientWidth)

theorem samplerRows_satisfied
    {rowsCount columns degree : Nat}
    (profile : PaperNifsPiRlcHandoff.SelectiveProfile rowsCount columns)
    (duplexBase : Nat) (constants : Poseidon2Schedule.Constants)
    (piCcsInput :
      KPiCcsTranscript.Input
        (PaperNifsPiRlcHandoff.SelectedShape profile) degree)
    (source : KPiRlcSemanticBinding.SourceColumns Params Arity 13)
    (quotients : PublicRole 13 → List Nat)
    (sourceValid : source.Valid)
    (quotientWidth : ∀ role, (quotients role).length = 53)
    (assignment : Nat → Nat)
    (satisfied :
      Satisfies
        (rows profile duplexBase constants piCcsInput source quotients
          sourceValid quotientWidth)
        assignment) :
    Satisfies
      (PaperNifsPiRlcHandoff.rows profile duplexBase constants piCcsInput)
      assignment :=
  fun row member => satisfied row (List.mem_append_left _ member)

theorem quotientRows_satisfied
    {rowsCount columns degree : Nat}
    (profile : PaperNifsPiRlcHandoff.SelectiveProfile rowsCount columns)
    (duplexBase : Nat) (constants : Poseidon2Schedule.Constants)
    (piCcsInput :
      KPiCcsTranscript.Input
        (PaperNifsPiRlcHandoff.SelectedShape profile) degree)
    (source : KPiRlcSemanticBinding.SourceColumns Params Arity 13)
    (quotients : PublicRole 13 → List Nat)
    (sourceValid : source.Valid)
    (quotientWidth : ∀ role, (quotients role).length = 53)
    (assignment : Nat → Nat)
    (satisfied :
      Satisfies
        (rows profile duplexBase constants piCcsInput source quotients
          sourceValid quotientWidth)
        assignment) :
    let input :=
      selectedProjectionInput profile duplexBase piCcsInput source quotients
    let valid :=
      projectionInput_valid duplexBase
        (KPiCcsTranscript.replay piCcsInput).afterOutput.lanes
        source quotients sourceValid quotientWidth
    Satisfies
      (KPiRlcProjectionTranscript.rows constants input valid)
      assignment := by
  dsimp only
  exact fun row member =>
    satisfied row (List.mem_append_right _ member)

/-- The quotient suffix's fresh builder denotes exactly the final sampler
state.  This is the physical transcript continuation theorem; no digest
equality or external state premise appears. -/
theorem decoded_selectedPrior_eq_samplerFinal
    {rowsCount columns degree : Nat}
    (profile : PaperNifsPiRlcHandoff.SelectiveProfile rowsCount columns)
    (duplexBase : Nat)
    (piCcsInput :
      KPiCcsTranscript.Input
        (PaperNifsPiRlcHandoff.SelectedShape profile) degree)
    (source : KPiRlcSemanticBinding.SourceColumns Params Arity 13)
    (quotients : PublicRole 13 → List Nat)
    (assignment : Nat → Nat) :
    SymbolicDuplexSemantics.decodedBuilder assignment
        (selectedProjectionInput profile duplexBase piCcsInput
          source quotients).prior =
      SymbolicDuplexSemantics.decodedBuilder assignment
        (PiRlcCanonicalSymbolicMachineHonest.fixedBuilder duplexBase
          (KPiCcsTranscript.replay piCcsInput).afterOutput.lanes) := by
  exact decoded_quotientPrior_eq_fixedBuilder _ _ _

/-! ## Semantic composition -/

/-- Every challenge list consumed by the quotient occurrence denotes the
centered first-accepted sampler output for the same typed paper source index.
The sampler-row satisfaction proof is derived by restriction from the
composed program. -/
def SamplerChallengesBound
    {rowsCount columns degree : Nat}
    (prime : EuclidPrime goldilocksP)
    (profile : PaperNifsPiRlcHandoff.SelectiveProfile rowsCount columns)
    (duplexBase : Nat) (constants : Poseidon2Schedule.Constants)
    (piCcsInput :
      KPiCcsTranscript.Input
        (PaperNifsPiRlcHandoff.SelectedShape profile) degree)
    (assignment : Nat → Nat)
    (residues : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (satisfied :
      Satisfies
        (PaperNifsPiRlcHandoff.rows
          profile duplexBase constants piCcsInput)
        assignment) : Prop :=
  ∀ index : Fin Arity.total,
    let lanes := (KPiCcsTranscript.replay piCcsInput).afterOutput.lanes
    let initial :=
      PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes
    let samplerSatisfied :=
      PaperNifsPiRlcHandoff.samplerRows_satisfied
        profile duplexBase constants piCcsInput assignment satisfied
    let u64Satisfied :=
      PiRlcCanonicalSamplerProgram.u64Rows_satisfied
        duplexBase constants lanes assignment samplerSatisfied
    (challengeColumns duplexBase index).map assignment =
      (PiRlcCanonicalSamplerSound.semanticOutput
        prime duplexBase
        (PiRlcCanonicalSamplerProgram.u64Base duplexBase)
        (PiRlcCanonicalSamplerProgram.candidateBase duplexBase)
        PiRlcCanonicalSamplerProgram.coordinateCount initial
        residues constantWire u64Satisfied
        (samplerCoordinate index)).map
          (fun coefficient =>
            (Phi81StrongSet.embedCoefficient coefficient).val)

theorem samplerChallengesBound_of_rows
    {rowsCount columns degree : Nat}
    (prime : EuclidPrime goldilocksP)
    (profile : PaperNifsPiRlcHandoff.SelectiveProfile rowsCount columns)
    (duplexBase : Nat) (constants : Poseidon2Schedule.Constants)
    (piCcsInput :
      KPiCcsTranscript.Input
        (PaperNifsPiRlcHandoff.SelectedShape profile) degree)
    (assignment : Nat → Nat)
    (residues : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (satisfied :
      Satisfies
        (PaperNifsPiRlcHandoff.rows
          profile duplexBase constants piCcsInput)
        assignment) :
    SamplerChallengesBound prime profile duplexBase constants piCcsInput
      assignment residues constantWire satisfied := by
  intro index
  have handoff :=
    PaperNifsPiRlcHandoff.rows_bind_sampler_to_piCcs
      prime profile duplexBase constants piCcsInput assignment
      residues constantWire satisfied (samplerCoordinate index)
  exact (challengeColumns_values duplexBase index assignment).trans handoff.2

/-- The sampler binding includes both the PiCCS outgoing state and every
centered coefficient vector. -/
theorem samplerBinding_of_rows
    {rowsCount columns degree : Nat}
    (prime : EuclidPrime goldilocksP)
    (profile : PaperNifsPiRlcHandoff.SelectiveProfile rowsCount columns)
    (duplexBase : Nat) (constants : Poseidon2Schedule.Constants)
    (piCcsInput :
      KPiCcsTranscript.Input
        (PaperNifsPiRlcHandoff.SelectedShape profile) degree)
    (assignment : Nat → Nat)
    (residues : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (satisfied :
      Satisfies
        (PaperNifsPiRlcHandoff.rows
          profile duplexBase constants piCcsInput)
        assignment) :
    SymbolicDuplexSemantics.decodedBuilder assignment
          (PiRlcCanonicalSymbolicMachineHonest.initialBuilder
            (KPiCcsTranscript.replay piCcsInput).afterOutput.lanes) =
        (KPiCcsTranscriptSemantics.valueReplay
          constants assignment piCcsInput).afterOutput
      ∧
        SamplerChallengesBound prime profile duplexBase constants piCcsInput
          assignment residues constantWire satisfied := by
  let first : Fin PiRlcCanonicalSamplerProgram.coordinateCount :=
    ⟨0, by decide⟩
  have handoff :=
    PaperNifsPiRlcHandoff.rows_bind_sampler_to_piCcs
      prime profile duplexBase constants piCcsInput assignment
      residues constantWire satisfied first
  exact
    ⟨handoff.1,
      samplerChallengesBound_of_rows prime profile duplexBase constants
        piCcsInput assignment residues constantWire satisfied⟩

/-- Satisfaction of the complete composed rows yields the paper `Pi_RLC`
equations over the physical sampler challenges, or the exact occurrence-bound
bad-root event at the transcript-derived quotient point. -/
theorem equations_or_transcriptBadRoot_of_rows
    {Assignment : Type}
    {semantics :
      RelationSemantics Unit Assignment PackedPublicInput
        FPrimeFullHistoryNifsPaper.Point
        FPrimeFullHistoryNifsPaper.Evaluation
        PackedCommitment}
    {rowsCount columns degree : Nat}
    (algebra :
      PiRLC.Algebra Unit Assignment PackedPublicInput
        FPrimeFullHistoryNifsPaper.Point
        FPrimeFullHistoryNifsPaper.Evaluation
        PackedCommitment Ring semantics Params)
    (codec : CarrierCodec 13)
    (ring : RingAlgebra)
    (algebraRefinement : AlgebraRefinement algebra codec ring)
    (profile : PaperNifsPiRlcHandoff.SelectiveProfile rowsCount columns)
    (duplexBase : Nat) (constants : Poseidon2Schedule.Constants)
    (piCcsInput :
      KPiCcsTranscript.Input
        (PaperNifsPiRlcHandoff.SelectedShape profile) degree)
    (source : KPiRlcSemanticBinding.SourceColumns Params Arity 13)
    (quotients : PublicRole 13 → List Nat)
    (sourceValid : source.Valid)
    (quotientWidth : ∀ role, (quotients role).length = 53)
    (assignment : Nat → Nat)
    (residues : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (satisfied :
      Satisfies
        (rows profile duplexBase constants piCcsInput source quotients
          sourceValid quotientWidth)
        assignment) :
    let input :=
      selectedProjectionInput profile duplexBase piCcsInput source quotients
    let valid :=
      projectionInput_valid duplexBase
        (KPiCcsTranscript.replay piCcsInput).afterOutput.lanes
        source quotients sourceValid quotientWidth
    PiRLC.Equations algebra
        (attempt codec assignment input.source.toBatchColumns) ∨
      (((KPiRlcProjectionTranscript.projectionColumns input).occurrence valid
          (KPiRlcProjectionTranscript.projectionBase input)).BadRoot
          assignment ∧
        (KPiRlcProjectionTranscript.betaColumns input).value assignment =
          (SymbolicDuplexSemantics.squeezeKValue constants
            (SymbolicDuplexSemantics.decodedBuilder assignment
              (KPiRlcProjectionTranscript.absorbed input))).1) := by
  dsimp only
  exact
    KPiRlcProjectionTranscript.equations_or_transcriptBadRoot_of_rows
      algebra codec ring algebraRefinement constants
      (selectedProjectionInput profile duplexBase piCcsInput source quotients)
      (projectionInput_valid duplexBase
        (KPiCcsTranscript.replay piCcsInput).afterOutput.lanes
        source quotients sourceValid quotientWidth)
      assignment residues constantWire
      (quotientRows_satisfied profile duplexBase constants piCcsInput
        source quotients sourceValid quotientWidth assignment satisfied)

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcQuotientHandoff

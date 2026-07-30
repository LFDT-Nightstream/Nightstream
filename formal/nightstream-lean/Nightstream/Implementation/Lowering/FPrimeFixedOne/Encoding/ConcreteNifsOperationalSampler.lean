import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalSelected
import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerCheckerRefinement

/-!
Contract: append the Lean-owned fixed-active ΠRLC sampler to one selected
operational ΠCCS call-frame occurrence.

The sampler starts from the symbolic post-output ΠCCS lanes at cursor one.
Its 15×54 output coordinates are bound by explicit equality rows to direct
proof-codec projections of `Certificate.piRlcChallenges`.  No carried
challenge, transcript state, or sampler acceptance is accepted as a premise.

This module does not yet own ΠDEC, activation, output materialization, or the
complete `nifsVerify` `CallRecipe`.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1200000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalSampler

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallColumnMap
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsGlobalColumnMap
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalProfile
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalOccurrence
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalSelected
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

private abbrev TranscriptState := Poseidon2Duplex.State

section SelectedFrame

variable {shape : SemanticShape}
variable {Digest AppState Witness Encoded RunningWitness FreshWitness : Type}
variable [DecidableEq AppState] [DecidableEq Encoded]
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}
variable {keys : Fin 1 →
  SelectedKey shape TranscriptState publicRingColumns publicFits verifierRows}
variable {defaultRunning :
  SelectedRunning shape publicRingColumns publicFits verifierRows}
variable {machine :
  Machine
    (SelectedKey shape TranscriptState publicRingColumns publicFits
      verifierRows)
    Digest AppState Witness
    (SelectedRunning shape publicRingColumns publicFits verifierRows)
    (SelectedFresh shape publicRingColumns publicFits verifierRows)
    Encoded 1}
variable {terminalRelations :
  TerminalRelations
    (SelectedKey shape TranscriptState publicRingColumns publicFits
      verifierRows)
    (SelectedRunning shape publicRingColumns publicFits verifierRows)
    RunningWitness
    (SelectedFresh shape publicRingColumns publicFits verifierRows)
    FreshWitness 1}
variable {terminalChecks :
  Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
    terminalRelations}
variable {widths : Widths} {footprints : Footprints}

local notation "Selected" =>
  ConcreteNifsParameters.selected keys defaultRunning machine
    terminalRelations terminalChecks widths footprints

private abbrev FamilyFor (application : Poseidon23ApplicationProfile Selected) :=
  application.family Selected

/-- First free numeric column after every ΠCCS transcript, claimed-chain, and
endpoint auxiliary. -/
def samplerBase
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {schema : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) schema (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) schema (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) schema (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil)))) : Nat :=
  let operationalInput :=
    ConcreteNifsOperationalOccurrence.input application profile frame
  KSplitNcOperationalRows.afterAllocation operationalInput

/-- The exact symbolic post-output ΠCCS state handed to ΠRLC. -/
def samplerLanes
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {schema : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) schema (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) schema (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) schema (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil)))) :
    Poseidon2Core.State :=
  (KSplitNcTranscript.outputBuilder
    (ConcreteNifsOperationalOccurrence.transcriptInput
      application profile frame)).lanes

/-- Exact fixed-active ΠRLC sampler rows, rooted at the ΠCCS output state. -/
def samplerRows
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {schema : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) schema (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) schema (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) schema (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil)))) :
    List Nightstream.Implementation.R1CS.Row :=
  PiRlcCanonicalSamplerProgram.rows
    (samplerBase application profile frame)
    profile.constants
    (samplerLanes application profile frame)

/-- Definitionally equal physical sampler coordinate. -/
def samplerCoordinate
    (coordinate :
      Fin
        Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.arity.total) :
    Fin PiRlcCanonicalSamplerProgram.coordinateCount :=
  ⟨coordinate.val, by
    simpa [PiRlcCanonicalSamplerProgram.coordinateCount] using
      coordinate.isLt⟩

/-- Definitionally equal physical selector position. -/
def samplerPosition (position : Fin ringDegree) :
    Fin PiRlcCanonicalSelector.outputCount :=
  ⟨position.val, by
    simpa [PiRlcCanonicalSelector.outputCount, ringDegree] using
      position.isLt⟩

/-- Direct proof-codec location of one carried challenge coordinate. -/
def challengeLocation
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {schema : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) schema (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) schema (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) schema (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (coordinate :
      Fin
        Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.arity.total)
    (position : Fin ringDegree) :
    FLocation (columnMap frame)
      ((profile.samplerViews.challenge coordinate position).column
        (proofOperand frame.operands) (proof_widthsAgree frame)) :=
  ConcreteNifsOperationalOccurrence.proofFieldLocation
    (FamilyFor application) frame
    (profile.samplerViews.challenge coordinate position)

/-- One explicit equality between a computed selector output and the
corresponding carried proof coordinate. -/
def challengeRow
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {schema : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) schema (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) schema (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) schema (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (coordinate :
      Fin
        Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.arity.total)
    (position : Fin ringDegree) :
    Nightstream.Implementation.R1CS.Row :=
  KEquality.equalityRow
    [(PiRlcCanonicalSelector.outputColumn
      (PiRlcCanonicalSamplerProgram.selectorBase
        (samplerBase application profile frame))
      (samplerCoordinate coordinate) (samplerPosition position), 1)]
    (challengeLocation application profile frame coordinate position).carried

/-- All 15×54 challenge-binding rows in coordinate-major order. -/
def challengeRows
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {schema : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) schema (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) schema (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) schema (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil)))) :
    List Nightstream.Implementation.R1CS.Row :=
  (List.ofFn fun coordinate =>
    List.ofFn fun position =>
      challengeRow application profile frame coordinate position).flatten

theorem challengeRows_length
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {schema : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) schema (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) schema (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) schema (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil)))) :
    (challengeRows application profile frame).length =
      Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.arity.total *
        ringDegree := by
  simp [challengeRows]
  omega

/-- Challenge-binding equations allocate no columns of their own. -/
def challengeCost : Cost where
  recurringRows :=
    Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.arity.total *
      ringDegree
  committedColumns := 0
  publicColumns := 0
  auxiliaryColumns := 0

/-- Exact cost of the operational ΠCCS and fixed-active ΠRLC sampler prefix. -/
def cost
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {schema : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) schema (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) schema (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) schema (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil)))) : Cost :=
  KSplitNcOperationalRows.cost
      (ConcreteNifsOperationalOccurrence.input application profile frame) +
    PiRlcCanonicalSamplerProgram.cost + challengeCost

/-- Exact operational ΠCCS plus ΠRLC sampler prefix. -/
def rows
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {schema : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) schema (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) schema (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) schema (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil)))) :
    List Nightstream.Implementation.R1CS.Row :=
  ConcreteNifsOperationalOccurrence.rows application profile frame ++
    samplerRows application profile frame ++
      challengeRows application profile frame

theorem rows_cost
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {schema : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) schema (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) schema (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) schema (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil)))) :
    (rows application profile frame).length =
      (cost application profile frame).recurringRows := by
  unfold rows cost challengeCost
    ConcreteNifsOperationalOccurrence.rows samplerRows
  rw [List.length_append, List.length_append]
  simp only [Cost.add_recurringRows]
  rw [KSplitNcOperationalRows.rows_cost,
    PiRlcCanonicalSamplerProgram.rows_length,
    challengeRows_length]

/-- Exact ordered auxiliary allocation for the prefix.  Challenge-binding
rows are equality-only reads and therefore contribute no entries. -/
def allocation
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {schema : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) schema (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) schema (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) schema (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil)))) : List Nat :=
  KSplitNcOperationalRows.columns
      (ConcreteNifsOperationalOccurrence.input application profile frame) ++
    PiRlcCanonicalSamplerProgram.allocation
      (samplerBase application profile frame)

theorem allocation_cost
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {schema : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) schema (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) schema (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) schema (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil)))) :
    (allocation application profile frame).length =
      (cost application profile frame).auxiliaryColumns := by
  unfold allocation cost challengeCost
  rw [List.length_append, KSplitNcOperationalRows.columns_length,
    KSplitNcOperationalRows.allocationWidth_eq_cost,
    PiRlcCanonicalSamplerProgram.allocation_length]
  simp only [Cost.add_auxiliaryColumns, Nat.add_zero]

theorem allocation_nodup
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {schema : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) schema (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) schema (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) schema (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil)))) :
    (allocation application profile frame).Nodup := by
  unfold allocation
  rw [List.nodup_append]
  refine
    ⟨KSplitNcOperationalRows.columns_nodup _,
      PiRlcCanonicalSamplerProgram.allocation_nodup _, ?_⟩
  intro left leftMember right rightMember equal
  subst right
  have below :=
    KSplitNcOperationalRows.columns_lt_afterAllocation _ left leftMember
  have above :=
    PiRlcCanonicalSamplerProgram.allocation_ge
      (samplerBase application profile frame) left rightMember
  exact (Nat.not_le_of_gt (by simpa [samplerBase] using below)) above

theorem piCcsRows_satisfied
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {schema : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) schema (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) schema (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) schema (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (assignment : Nat → Nat)
    (satisfied : Satisfies (rows application profile frame) assignment) :
    Satisfies
      (ConcreteNifsOperationalOccurrence.rows application profile frame)
      assignment :=
  fun row member =>
    satisfied row (List.mem_append_left _ (List.mem_append_left _ member))

theorem samplerRows_satisfied
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {schema : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) schema (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) schema (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) schema (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (assignment : Nat → Nat)
    (satisfied : Satisfies (rows application profile frame) assignment) :
    Satisfies (samplerRows application profile frame) assignment :=
  fun row member =>
    satisfied row
      (List.mem_append_left _
        (List.mem_append_right _ member))

theorem challengeRows_satisfied
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {schema : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) schema (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) schema (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) schema (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (assignment : Nat → Nat)
    (satisfied : Satisfies (rows application profile frame) assignment) :
    Satisfies (challengeRows application profile frame) assignment :=
  fun row member =>
    satisfied row (List.mem_append_right _ member)

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalSampler

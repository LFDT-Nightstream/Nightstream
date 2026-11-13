import NightstreamFPrime.Lifecycle.BoundedDepth.Keys
import NightstreamFPrime.Lifecycle.RandomOracle.Schedule

/-! The refresh fold's transcript as a random-oracle `Schedule` (`schedule`), and the refresh
key's reads of it. The refresh transcript absorbs the fresh claim and the commitments and
evaluation families of the Π_DEC children (`BoundedDepth.refreshWords`). So its statement is the
fresh claim with the children split of the refresh proof (`Statement`, `statement`). The
refresh key's Π_CCS coins (`coins_eq_reads`, `piCcsProbe_coins`) and its 17 Π_RLC challenges
(`rho_seal`) are the schedule's reads. This is `TranscriptCoverage` for `BoundedDepth.refreshKey`.
The words of the fresh claim (`freshWords_injective`) are also the light statement's words. -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.RefreshSchedule

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.BoundedDepth
open NightstreamFPrime.Lifecycle.RandomOracle
open NightstreamFPrime.Lifecycle.RandomOracle.Sponge

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ RingCarrierLayout.carrierWidth logicalWidth}

section Words

theorem freshWords_eq (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (ProductionKey.publicInputBlocks fresh).flatten =
      ProductionKey.priorDigest fresh ++
        (List.finRange productionShape.freshCount).flatMap fun index =>
          serializeCommitment (fresh.commitments index) ++
            serializePublicInput (fresh.publicInputs index) := by
  simp [ProductionKey.publicInputBlocks, List.flatMap, List.flatten_flatten, List.map_map,
    Function.comp_def]

theorem freshWords_length
    (fresh fresh' : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (ProductionKey.publicInputBlocks fresh).flatten.length =
      (ProductionKey.publicInputBlocks fresh').flatten.length := by
  rw [freshWords_eq, freshWords_eq, List.length_append, List.length_append,
    flatMap_length_eq _ _
      (fun index => serializeCommitment (fresh'.commitments index) ++
        serializePublicInput (fresh'.publicInputs index))
      (fun _ _ => by simp)]
  simp [ProductionKey.priorDigest, decodeHash]

/-- The words of the fresh claim fix the fresh claim. -/
theorem freshWords_injective
    {fresh fresh' : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)}
    (same : (ProductionKey.publicInputBlocks fresh).flatten =
      (ProductionKey.publicInputBlocks fresh').flatten) : fresh = fresh' := by
  rw [freshWords_eq, freshWords_eq] at same
  obtain ⟨_, perFresh⟩ := List.append_inj same (by simp [ProductionKey.priorDigest, decodeHash])
  have each := flatMap_eq_of_lengths _ _ _ (fun _ _ => by simp) perFresh
  have parts := fun index => List.append_inj (each index (List.mem_finRange index)) (by simp)
  have commitments : fresh.commitments = fresh'.commitments := by
    funext index
    exact serializeCommitment_injective (parts index).1
  have publicInputs : fresh.publicInputs = fresh'.publicInputs := by
    funext index
    exact serializePublicInput_injective (parts index).2
  cases fresh
  cases fresh'
  simp only at commitments publicInputs
  rw [commitments, publicInputs]

/-- Each Π_DEC child's commitment and evaluation family, as `BoundedDepth.childWords` reads them. -/
def splitWords (split : Refresh.Split PaperAlgebra.Commitment productionShape) : List F :=
  (List.finRange productionShape.runningCount).flatMap fun child =>
    serializeCommitment (split.commitments child) ++ serializeEvaluations (split.evaluations child)

theorem splitWords_length (split split' : Refresh.Split PaperAlgebra.Commitment productionShape) :
    (splitWords split).length = (splitWords split').length :=
  flatMap_length_eq _ _ _ fun _ _ => by simp

theorem splitWords_injective {split split' : Refresh.Split PaperAlgebra.Commitment productionShape}
    (same : splitWords split = splitWords split') : split = split' := by
  have blocks := flatMap_eq_of_lengths _ _ _ (fun _ _ => by simp) same
  have parts := fun child => List.append_inj (blocks child (List.mem_finRange child)) (by simp)
  have commitments : split.commitments = split'.commitments :=
    funext fun child => serializeCommitment_injective (parts child).1
  have evaluations : split.evaluations = split'.evaluations :=
    funext fun child => serializeEvaluations_injective (parts child).2
  cases split
  cases split'
  simp only at commitments evaluations
  rw [commitments, evaluations]

end Words

section Schedule

variable (logicalWidth publicFits) in
/-- What the refresh transcript absorbs before the Π_CCS coins: the fresh claim and the
children split. -/
abbrev Statement :=
  Fresh (logicalWidth := logicalWidth) (publicFits := publicFits) ×
    Refresh.Split PaperAlgebra.Commitment productionShape

def statementWords (statement : Statement logicalWidth publicFits) : List F :=
  (ProductionKey.publicInputBlocks statement.1).flatten ++ splitWords statement.2

theorem statementWords_length (statement statement' : Statement logicalWidth publicFits) :
    (statementWords statement).length = (statementWords statement').length := by
  rw [statementWords, statementWords, List.length_append, List.length_append,
    freshWords_length statement.1 statement'.1, splitWords_length statement.2 statement'.2]

theorem statementWords_injective {statement statement' : Statement logicalWidth publicFits}
    (same : statementWords statement = statementWords statement') : statement = statement' := by
  obtain ⟨fresh, split⟩ := List.append_inj same (freshWords_length statement.1 statement'.1)
  exact Prod.ext (freshWords_injective fresh) (splitWords_injective split)

theorem fullOutputWords_eq (output : FullOutputCoordinates.FullOutput K productionShape) :
    ProductionKey.fullOutputWords output = Schedule.fullOutputWords output :=
  rfl

variable (logicalWidth publicFits) in
/-- The refresh transcript's schedule: production shape, the refresh domain, and 17 Π_RLC
challenges. -/
def schedule : Schedule where
  shape := productionShape
  domainChunk := refreshFoldDomainChunk
  domainChunk_length := by simp [refreshFoldDomainChunk, domainChunk]
  Statement := Statement logicalWidth publicFits
  statementWords := statementWords
  statementWords_length := statementWords_length
  statementWords_injective := statementWords_injective
  Output := FullOutputCoordinates.FullOutput K productionShape
  outputWords := ProductionKey.fullOutputWords
  outputWords_length := Schedule.fullOutputWords_length
  outputWords_injective := Schedule.fullOutputWords_injective
  rhoCount := (Nifs.PaperProfile.arity).total
  rhoCount_le := by decide

/-- The refresh statement of a Π_CCS run on the children `running`. -/
def statement (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (schedule logicalWidth publicFits).Statement :=
  (fresh, ⟨running.commitments, running.evaluations⟩)

def messages {degree : Nat} (proof : Proof degree) :
    (schedule logicalWidth publicFits).Messages degree :=
  ⟨proof.piCcsRounds, proof.piCcsOutput⟩

end Schedule

section Seals

variable (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
  (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
  (proof : Proof (ProductionKey.degreeBound relation))

theorem publicInputState_eq_run :
    (refreshKey relation ajtai).publicInputState running fresh =
      run Transcript.initialState
        ((schedule logicalWidth publicFits).statementCalls (statement running fresh)) := by
  show Transcript.absorb refreshFoldInitialState (refreshWords running fresh) = _
  rw [refreshFoldInitialState, absorb_eq_run, absorb_eq_run, Schedule.statementCalls, run_append]
  rfl

theorem coins_eq_reads :
    ((refreshKey relation ajtai).piCcsExecution running fresh proof).coins =
      { alpha := ⟨List.ofFn fun coordinate => readK (slotOf coordinate.val)
            (run Transcript.initialState ((schedule logicalWidth publicFits).challengeCalls
              (statement running fresh) (messages proof) (.alpha coordinate))),
          List.length_ofFn⟩
        gamma := readK (slotOf productionShape.cubeVariables) (run Transcript.initialState
          ((schedule logicalWidth publicFits).challengeCalls (statement running fresh)
            (messages proof) .gamma))
        roundPoint := ⟨List.ofFn fun index => readK (slotOf 0) (run Transcript.initialState
            ((schedule logicalWidth publicFits).challengeCalls (statement running fresh)
              (messages proof) (.round index))),
          List.length_ofFn⟩
        finalState := run Transcript.initialState
          ((schedule logicalWidth publicFits).roundsCalls (statement running fresh)
            (messages proof)) } := by
  rw [Nifs.PaperNonInteractive.Key.piCcsExecution_coins_eq_derive]
  exact Schedule.derive_eq_reads (schedule logicalWidth publicFits) (statement running fresh)
    (messages proof) (publicInputState_eq_run relation ajtai running fresh)

theorem piCcsProbe_coins :
    ((refreshKey relation ajtai).piCcsProbe running fresh proof).coins =
      (schedule logicalWidth publicFits).coinsFrom Schedule.spongeRead (statement running fresh)
        (messages proof) := by
  simp only [Nifs.PaperNonInteractive.Key.piCcsProbe,
    coins_eq_reads relation ajtai running fresh proof]
  rfl

private theorem outgoingState_eq_run :
    ((refreshKey relation ajtai).piCcsExecution running fresh proof).outgoingState =
      run Transcript.initialState
        ((schedule logicalWidth publicFits).outputCalls (statement running fresh)
          (messages proof)) := by
  rw [Nifs.PaperNonInteractive.Key.piCcsExecution_outgoingState_eq_absorbPiCcsOutput,
    coins_eq_reads relation ajtai running fresh proof]
  exact Schedule.absorbOutput_eq_run (schedule logicalWidth publicFits) (statement running fresh)
    (messages proof)

theorem rho_seal :
    (refreshKey relation ajtai).piRlcChallenges running fresh proof =
      some fun index => readRho (run Transcript.initialState
        ((schedule logicalWidth publicFits).challengeCalls (statement running fresh)
          (messages proof) (.rho index))) := by
  unfold Nifs.PaperNonInteractive.Key.piRlcChallenges
  show ProductionKey.piRlcResponse
      ((refreshKey relation ajtai).piCcsExecution running fresh proof).outgoingState = _
  rw [ProductionKey.piRlcResponse, Transcript.PiRlcSampler.piRlcChallenges,
    Transcript.PiRlcSampler.piRlcChallengesWithState_challenges, outgoingState_eq_run]
  exact congrArg some (funext fun index => Schedule.rho_read (schedule logicalWidth publicFits)
    (statement running fresh) (messages proof) index)

end Seals

end NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.RefreshSchedule

import NightstreamFPrime.Lifecycle.ProductionKey
import NightstreamFPrime.Spec.Folding.PiCCS.Transcript
import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.TranscriptHistory

namespace NightstreamFPrime.Lifecycle.TranscriptCoverage

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

abbrev Call := Fin Poseidon2.rate → F

def squeeze : Call := fun _ => 0

def step (state : Transcript.State) (chunk : Call) : Transcript.State :=
  Poseidon2.absorbBlock state (List.ofFn chunk)

def run (state : Transcript.State) (calls : List Call) : Transcript.State :=
  calls.foldl step state

@[simp] private theorem run_nil (state : Transcript.State) : run state [] = state := rfl

@[simp] private theorem run_cons (state : Transcript.State) (call : Call) (calls : List Call) :
    run state (call :: calls) = run (step state call) calls := rfl

private theorem run_append (state : Transcript.State) (left right : List Call) :
    run state (left ++ right) = run (run state left) right := by
  simp [run, List.foldl_append]

def padChunk (words : List F) : Call :=
  fun lane => words.getD lane.val 0

private theorem absorbBlock_padChunk (state : Transcript.State) {words : List F}
    (short : words.length ≤ Poseidon2.rate) :
    Poseidon2.absorbBlock state words =
      Poseidon2.absorbBlock state (List.ofFn (padChunk words)) := by
  unfold Poseidon2.absorbBlock
  congr 1
  apply List.map_congr_left
  intro lane _
  congr 1
  by_cases inside : lane < Poseidon2.rate
  · simp [padChunk, List.getD_eq_getElem?_getD, inside]
  · have outside : Poseidon2.rate ≤ lane := Nat.le_of_not_lt inside
    rw [List.getD_eq_default _ _ (Nat.le_trans short outside),
      List.getD_eq_default _ _ (by simpa using outside)]

def absorbCalls (words : List F) : List Call :=
  (List.range ((words.length + Poseidon2.rate - 1) / Poseidon2.rate)).map fun chunk =>
    padChunk ((words.drop (chunk * Poseidon2.rate)).take Poseidon2.rate)

private theorem absorb_eq_run (state : Transcript.State) (words : List F) :
    Transcript.absorb state words = run state (absorbCalls words) := by
  unfold Transcript.absorb absorbCalls run
  rw [List.foldl_map, List.foldl_map]
  apply List.foldl_ext
  intro current chunk _
  exact absorbBlock_padChunk current (List.length_take_le _ _)

@[simp] private theorem absorbCalls_length (words : List F) :
    (absorbCalls words).length = (words.length + Poseidon2.rate - 1) / Poseidon2.rate := by
  simp [absorbCalls]

private theorem absorbCalls_ne_nil {words : List F} (nonempty : words ≠ []) :
    absorbCalls words ≠ [] := by
  intro empty
  have length := congrArg List.length empty
  have positive : 0 < words.length := List.length_pos_iff.mpr nonempty
  simp only [absorbCalls_length, List.length_nil] at length
  unfold Poseidon2.rate at length
  omega

private theorem absorbCalls_injective {left right : List F}
    (lengthEqual : left.length = right.length)
    (same : absorbCalls left = absorbCalls right) : left = right := by
  have chunks : ∀ chunk ∈ List.range ((left.length + Poseidon2.rate - 1) / Poseidon2.rate),
      padChunk ((left.drop (chunk * Poseidon2.rate)).take Poseidon2.rate) =
        padChunk ((right.drop (chunk * Poseidon2.rate)).take Poseidon2.rate) := by
    unfold absorbCalls at same
    rw [← lengthEqual] at same
    exact List.map_inj_left.mp same
  apply List.ext_getElem lengthEqual
  intro position leftBound rightBound
  have chunkBound : position / Poseidon2.rate <
      (left.length + Poseidon2.rate - 1) / Poseidon2.rate := by
    unfold Poseidon2.rate at *
    omega
  have lane := congrFun (chunks (position / Poseidon2.rate) (List.mem_range.mpr chunkBound))
    ⟨position % Poseidon2.rate, Nat.mod_lt _ (by decide)⟩
  have split : position / Poseidon2.rate * Poseidon2.rate +
      position % Poseidon2.rate = position := by
    rw [Nat.mul_comm]
    exact Nat.div_add_mod position Poseidon2.rate
  simp only [padChunk, List.getD_eq_getElem?_getD, List.getElem?_take,
    Nat.mod_lt _ (show 0 < Poseidon2.rate by decide), ite_true,
    List.getElem?_drop, split] at lane
  simpa [List.getElem?_eq_getElem leftBound, List.getElem?_eq_getElem rightBound]
    using lane

abbrev Slot := Fin Transcript.coinsPerChunk

def slotOf (position : Nat) : Slot :=
  ⟨position % Transcript.coinsPerChunk, Nat.mod_lt _ (by decide)⟩

def readK (slot : Slot) (state : Transcript.State) : K :=
  ⟨state.getD (2 * slot.val) 0, state.getD (2 * slot.val + 1) 0⟩

private theorem absorbCalls_zeroChunk : absorbCalls Transcript.zeroChunk = [squeeze] := by
  have chunk : padChunk ((Transcript.zeroChunk.drop (0 * Poseidon2.rate)).take Poseidon2.rate) =
      squeeze := by
    funext lane
    simp [padChunk, squeeze, Transcript.zeroChunk, List.getD_eq_getElem?_getD]
  unfold absorbCalls
  rw [show (Transcript.zeroChunk.length + Poseidon2.rate - 1) / Poseidon2.rate = 1 from rfl,
    List.range_one, List.map_cons, List.map_nil, chunk]

def refresh (position : Nat) : List Call :=
  if position % Transcript.coinsPerChunk = Transcript.coinsPerChunk - 1 then [squeeze] else []

private theorem replicate_refresh (position : Nat) :
    List.replicate (position / Transcript.coinsPerChunk) squeeze ++ refresh position =
      List.replicate ((position + 1) / Transcript.coinsPerChunk) squeeze := by
  have six : Transcript.coinsPerChunk = 6 := rfl
  unfold refresh
  split
  · rename_i last
    rw [six] at last ⊢
    rw [show (position + 1) / 6 = position / 6 + 1 by omega, List.replicate_succ']
  · rename_i notLast
    rw [six] at notLast ⊢
    rw [show (position + 1) / 6 = position / 6 by omega, List.append_nil]

private theorem oracle_squeeze_eq (state : Transcript.State)
    (label : FiatShamir.ChallengeLabel productionShape) :
    Transcript.piCcsOracle.transcript.squeeze state label =
      (readK (slotOf (Transcript.coinPosition label)) state,
        run state (refresh (Transcript.coinPosition label))) := by
  show Transcript.squeezeAt state (Transcript.coinPosition label) = _
  by_cases last : Transcript.coinPosition label % Transcript.coinsPerChunk =
      Transcript.coinsPerChunk - 1
  · rw [Transcript.squeezeAt, ite_eq_left last, refresh, ite_eq_left last, absorb_eq_run,
      absorbCalls_zeroChunk]
    rfl
  · rw [Transcript.squeezeAt, ite_eq_right last, refresh, ite_eq_right last]
    rfl

private theorem squeezeMany_run (state : Transcript.State) (start : Nat)
    (labels : List (FiatShamir.ChallengeLabel productionShape))
    (positions : ∀ index (inside : index < labels.length),
      Transcript.coinPosition labels[index] = start + index) :
    (FiatShamir.squeezeMany Transcript.piCcsOracle.transcript
        (run state (List.replicate (start / Transcript.coinsPerChunk) squeeze)) labels).2 =
      run state (List.replicate ((start + labels.length) / Transcript.coinsPerChunk) squeeze) ∧
    ∀ index, index < labels.length →
      (FiatShamir.squeezeMany Transcript.piCcsOracle.transcript
          (run state (List.replicate (start / Transcript.coinsPerChunk) squeeze)) labels).1.getD
          index K.zero =
        readK (slotOf (start + index))
          (run state (List.replicate ((start + index) / Transcript.coinsPerChunk) squeeze)) := by
  induction labels generalizing start with
  | nil => exact ⟨by simp [FiatShamir.squeezeMany], fun index inside => absurd inside (by simp)⟩
  | cons label labels inductionHypothesis =>
      have first := positions 0 (by simp)
      simp only [List.getElem_cons_zero, Nat.add_zero] at first
      have rest := inductionHypothesis (start + 1) fun index inside => by
        have shifted := positions (index + 1) (by simpa using inside)
        simp only [List.getElem_cons_succ] at shifted
        rw [shifted]
        omega
      have next : run (run state (List.replicate (start / Transcript.coinsPerChunk) squeeze))
          (refresh start) =
          run state (List.replicate ((start + 1) / Transcript.coinsPerChunk) squeeze) := by
        rw [← run_append, replicate_refresh]
      simp only [FiatShamir.squeezeMany, oracle_squeeze_eq, first, next]
      refine ⟨?_, fun index inside => ?_⟩
      · rw [rest.1, List.length_cons, Nat.add_assoc, Nat.add_comm 1]
      · cases index with
        | zero => simp
        | succ index =>
            rw [List.getD_cons_succ, rest.2 index (by simpa using inside),
              Nat.add_assoc, Nat.add_comm 1]

def messages {degree : Nat} (proof : Proof degree) :
    Fin productionShape.cubeVariables → SumCheck.Finite.Message K :=
  fun round => (proof.piCcsRounds round).toMessage

def messageCalls (rounds : Fin productionShape.cubeVariables → SumCheck.Finite.Message K)
    (round : Fin productionShape.cubeVariables) : List Call :=
  absorbCalls (Transcript.serializeMessage (rounds round))

private theorem oracle_absorbRound_eq (state : Transcript.State)
    (rounds : Fin productionShape.cubeVariables → SumCheck.Finite.Message K)
    (round : Fin productionShape.cubeVariables) :
    Transcript.piCcsOracle.transcript.absorbRound state round (rounds round) =
      run state (messageCalls rounds round) :=
  absorb_eq_run state _

private theorem oracle_round_squeeze_eq (state : Transcript.State)
    (round : Fin productionShape.cubeVariables) :
    Transcript.piCcsOracle.transcript.squeeze state (.sumcheck round) =
      (readK (slotOf 0) state, state) := by
  rw [oracle_squeeze_eq]
  rfl

private theorem deriveRoundsFrom_state
    (rounds : Fin productionShape.cubeVariables → SumCheck.Finite.Message K)
    (state : Transcript.State) (indices : List (Fin productionShape.cubeVariables)) :
    (FiatShamir.deriveRoundsFrom Transcript.piCcsOracle.transcript rounds state indices).2 =
      run state (indices.flatMap (messageCalls rounds)) := by
  induction indices generalizing state with
  | nil => rfl
  | cons round indices inductionHypothesis =>
      simp only [FiatShamir.deriveRoundsFrom, oracle_round_squeeze_eq, List.flatMap_cons,
        run_append, oracle_absorbRound_eq]
      rw [inductionHypothesis]

private theorem deriveRoundsFrom_getD
    (rounds : Fin productionShape.cubeVariables → SumCheck.Finite.Message K)
    (state : Transcript.State) (indices : List (Fin productionShape.cubeVariables))
    (index : Nat) (bound : index < indices.length) :
    (FiatShamir.deriveRoundsFrom Transcript.piCcsOracle.transcript rounds state
        indices).1.getD index K.zero =
      readK (slotOf 0) (run state ((indices.take index).flatMap (messageCalls rounds) ++
        messageCalls rounds indices[index])) := by
  induction indices generalizing state index with
  | nil => simp at bound
  | cons round indices inductionHypothesis =>
      cases index with
      | zero =>
          simp only [FiatShamir.deriveRoundsFrom, oracle_round_squeeze_eq, List.getD_cons_zero,
            List.take_zero, List.flatMap_nil, List.nil_append, List.getElem_cons_zero,
            oracle_absorbRound_eq]
      | succ index =>
          simp only [FiatShamir.deriveRoundsFrom, oracle_round_squeeze_eq, List.getD_cons_succ,
            List.take_succ_cons, List.flatMap_cons, List.getElem_cons_succ]
          rw [inductionHypothesis _ index (by simpa using bound)]
          simp only [run_append, List.append_assoc, oracle_absorbRound_eq]

section Schedule

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

def statementCalls
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)) : List Call :=
  absorbCalls Transcript.foldDomainChunk ++
    absorbCalls (ProductionKey.publicInputBlocks fresh).flatten

def preRoundCalls
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)) : List Call :=
  statementCalls fresh ++
    List.replicate ((productionShape.cubeVariables + 1) / Transcript.coinsPerChunk) squeeze

def roundPrefixCalls {degree : Nat}
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof degree) (round : Fin productionShape.cubeVariables) : List Call :=
  preRoundCalls fresh ++
    ((canonicalFinIndices productionShape.cubeVariables).take round.val).flatMap
      (messageCalls (messages proof)) ++
    messageCalls (messages proof) round

def outputCalls {degree : Nat}
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof degree) : List Call :=
  preRoundCalls fresh ++
    (canonicalFinIndices productionShape.cubeVariables).flatMap (messageCalls (messages proof)) ++
    absorbCalls (ProductionKey.fullOutputWords proof.piCcsOutput)

end Schedule

def widen (draw : Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Draw) : Call := fun lane =>
  if inside : lane.val < Spec.Folding.Nifs.NonInteractive.PiRlcSampler.drawWidth then
    draw ⟨lane.val, inside⟩
  else 0

def rhoIndex (index : Fin (Nifs.PaperProfile.arity).total) : Fin 17 :=
  ⟨index.val, index.isLt.trans_eq (by decide)⟩

def readRho (state : Transcript.State) : RingF :=
  Phi81StrongSet.embedScalar
    (Spec.Folding.Nifs.NonInteractive.PiRlcSampler.sample
      (Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.block state))

private theorem step_widen (state : Transcript.State)
    (draw : Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Draw) :
    step state (widen draw) = Poseidon2.absorbBlock state (List.ofFn draw) := by
  unfold step Poseidon2.absorbBlock
  congr 1

private theorem run_map_widen (state : Transcript.State)
    (history : List Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Draw) :
    run state (history.map widen) =
      Spec.Folding.Nifs.NonInteractive.PiRlcSampler.TranscriptHistory.replay state history := by
  induction history generalizing state with
  | nil => rfl
  | cons draw history inductionHypothesis =>
      show run (step state (widen draw)) (history.map widen) =
        Spec.Folding.Nifs.NonInteractive.PiRlcSampler.TranscriptHistory.replay
          (Poseidon2.absorbBlock state (List.ofFn draw)) history
      rw [step_widen]
      exact inductionHypothesis _

inductive Challenge where
  | alpha (coordinate : Fin productionShape.cubeVariables)
  | gamma
  | round (index : Fin productionShape.cubeVariables)
  | rho (index : Fin (Nifs.PaperProfile.arity).total)
  deriving DecidableEq, Fintype

abbrev Challenge.Value : Challenge → Type
  | .alpha _ => K
  | .gamma => K
  | .round _ => K
  | .rho _ => RingF

def Challenge.slot : Challenge → Slot
  | .alpha coordinate => slotOf coordinate.val
  | .gamma => slotOf productionShape.cubeVariables
  | .round _ => slotOf 0
  | .rho _ => slotOf 0

def Challenge.read : (challenge : Challenge) → Transcript.State → challenge.Value
  | .alpha coordinate => readK (slotOf coordinate.val)
  | .gamma => readK (slotOf productionShape.cubeVariables)
  | .round _ => readK (slotOf 0)
  | .rho _ => readRho

def fixedCalls : Challenge → List Call
  | .alpha coordinate => List.replicate (coordinate.val / Transcript.coinsPerChunk) squeeze
  | .gamma => List.replicate (productionShape.cubeVariables / Transcript.coinsPerChunk) squeeze
  | .round _ => []
  | .rho index =>
      (Spec.Folding.Nifs.NonInteractive.PiRlcSampler.ScheduleLaw.queryAt []
        (rhoIndex index)).val.map widen

section Contract

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

def proverCalls {degree : Nat}
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof degree) : Challenge → List Call
  | .alpha _ => statementCalls fresh
  | .gamma => statementCalls fresh
  | .round index => roundPrefixCalls fresh proof index
  | .rho _ => outputCalls fresh proof

def challengeCalls {degree : Nat}
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof degree) (challenge : Challenge) : List Call :=
  proverCalls fresh proof challenge ++ fixedCalls challenge

def challengeAddress {degree : Nat}
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof degree) (challenge : Challenge) : List Call × Slot :=
  (challengeCalls fresh proof challenge, challenge.slot)

def coinsFrom {degree : Nat} (read : List Call × Slot → K)
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof degree) : StrongReduction.PublicCoins K productionShape where
  alpha := ⟨List.ofFn fun coordinate => read (challengeAddress fresh proof (.alpha coordinate)),
    List.length_ofFn⟩
  gamma := read (challengeAddress fresh proof .gamma)
  roundPoint := ⟨List.ofFn fun index => read (challengeAddress fresh proof (.round index)),
    List.length_ofFn⟩

section Seals

variable (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
  (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
  (proof : Proof (ProductionKey.degreeBound relation))

noncomputable def keyChallenge :
    (challenge : Challenge) → Option challenge.Value
  | .alpha coordinate =>
      let alpha := ((ProductionKey.key relation ajtai).piCcsExecution running fresh
        proof).coins.alpha
      some (alpha.coordinates[coordinate.val]'(by rw [alpha.dimension]; exact coordinate.isLt))
  | .gamma => some ((ProductionKey.key relation ajtai).piCcsExecution running fresh
      proof).coins.gamma
  | .round index =>
      let point := ((ProductionKey.key relation ajtai).piCcsExecution running fresh
        proof).coins.roundPoint
      some (point.coordinates[index.val]'(by rw [point.dimension]; exact index.isLt))
  | .rho index => ((ProductionKey.key relation ajtai).piRlcChallenges running fresh
      proof).map fun challenges => challenges index

theorem publicInputState_eq_run :
    (ProductionKey.key relation ajtai).publicInputState running fresh =
      run Transcript.initialState (statementCalls fresh) := by
  rw [ProductionKey.key_publicInputState_eq, Transcript.foldInitialState, absorb_eq_run,
    absorb_eq_run, statementCalls, run_append]

private theorem alphaLabels_positions (index : Nat)
    (inside : index < (FiatShamir.alphaLabels productionShape).length) :
    Transcript.coinPosition (FiatShamir.alphaLabels productionShape)[index] = 0 + index := by
  simp [FiatShamir.alphaLabels, canonicalFinIndices, Transcript.coinPosition]

private theorem alphaSqueezes :
    FiatShamir.squeezeMany Transcript.piCcsOracle.transcript
        ((ProductionKey.key relation ajtai).publicInputState running fresh)
        (FiatShamir.alphaLabels productionShape) =
      FiatShamir.squeezeMany Transcript.piCcsOracle.transcript
        (run (run Transcript.initialState (statementCalls fresh))
          (List.replicate (0 / Transcript.coinsPerChunk) squeeze))
        (FiatShamir.alphaLabels productionShape) := by
  rw [publicInputState_eq_run, Nat.zero_div, List.replicate_zero, run_nil]

private theorem alpha_seal
    (coordinate : Fin productionShape.cubeVariables) :
    ((ProductionKey.key relation ajtai).piCcsExecution running fresh
        proof).coins.alpha.coordinates.getD coordinate.val K.zero =
      readK (slotOf coordinate.val) (run Transcript.initialState
        (statementCalls fresh ++ fixedCalls (.alpha coordinate))) := by
  rw [Nifs.PaperNonInteractive.Key.piCcsExecution_coins_eq_derive]
  change (FiatShamir.squeezeMany Transcript.piCcsOracle.transcript
      ((ProductionKey.key relation ajtai).publicInputState running fresh)
      (FiatShamir.alphaLabels productionShape)).1.getD coordinate.val K.zero = _
  rw [alphaSqueezes, ((squeezeMany_run _ 0 _ alphaLabels_positions).2 coordinate.val
    (by simp [FiatShamir.alphaLabels_length])), ← run_append]
  simp only [Nat.zero_add, fixedCalls]

private theorem alphaState_eq_run :
    (FiatShamir.squeezeMany Transcript.piCcsOracle.transcript
        ((ProductionKey.key relation ajtai).publicInputState running fresh)
        (FiatShamir.alphaLabels productionShape)).2 =
      run Transcript.initialState (statementCalls fresh ++ fixedCalls .gamma) := by
  rw [alphaSqueezes, (squeezeMany_run _ 0 _ alphaLabels_positions).1, ← run_append]
  simp only [Nat.zero_add, FiatShamir.alphaLabels_length, fixedCalls]

private theorem gamma_seal :
    ((ProductionKey.key relation ajtai).piCcsExecution running fresh proof).coins.gamma =
      readK (slotOf productionShape.cubeVariables)
        (run Transcript.initialState (statementCalls fresh ++ fixedCalls .gamma)) := by
  rw [Nifs.PaperNonInteractive.Key.piCcsExecution_coins_eq_derive]
  change (Transcript.piCcsOracle.transcript.squeeze
      (FiatShamir.squeezeMany Transcript.piCcsOracle.transcript
        ((ProductionKey.key relation ajtai).publicInputState running fresh)
        (FiatShamir.alphaLabels productionShape)).2 .gamma).1 = _
  rw [oracle_squeeze_eq, alphaState_eq_run]
  simp only [Transcript.coinPosition]

private theorem preRoundState_eq_run :
    (Transcript.piCcsOracle.transcript.squeeze
        (FiatShamir.squeezeMany Transcript.piCcsOracle.transcript
          ((ProductionKey.key relation ajtai).publicInputState running fresh)
          (FiatShamir.alphaLabels productionShape)).2 .gamma).2 =
      run Transcript.initialState (preRoundCalls fresh) := by
  rw [oracle_squeeze_eq, alphaState_eq_run, ← run_append, preRoundCalls, List.append_assoc]
  simp only [fixedCalls, Transcript.coinPosition, replicate_refresh]

private theorem round_seal
    (round : Fin productionShape.cubeVariables) :
    ((ProductionKey.key relation ajtai).piCcsExecution running fresh
        proof).coins.roundPoint.coordinates.getD round.val K.zero =
      readK (slotOf 0) (run Transcript.initialState
        (roundPrefixCalls fresh proof round ++ fixedCalls (.round round))) := by
  rewrite [Nifs.PaperNonInteractive.Key.piCcsExecution_coins_eq_derive,
    (PiCCS.Transcript.derive_rounds_holds _ _ _).roundPoint_eq,
    ← PiCCS.Transcript.deriveFromState_initialState,
    ProductionKey.key_oracle_eq, Transcript.piCcsOracle.initialState_is_prior]
  dsimp only [PiCCS.Transcript.deriveFromState]
  rw [deriveRoundsFrom_getD _ _ _ _ (by simp [canonicalFinIndices]),
    preRoundState_eq_run, ← run_append, roundPrefixCalls, canonicalFinIndices_getElem]
  simp only [fixedCalls, Fin.eta, List.append_assoc, List.append_nil]
  rfl

private theorem roundsState_eq_run :
    ((ProductionKey.key relation ajtai).piCcsExecution running fresh
        proof).coins.finalState =
      run Transcript.initialState (preRoundCalls fresh ++
        (canonicalFinIndices productionShape.cubeVariables).flatMap
          (messageCalls (messages proof))) := by
  unfold messages
  rewrite [Nifs.PaperNonInteractive.Key.piCcsExecution_coins_eq_derive,
    (PiCCS.Transcript.derive_rounds_holds _ _ _).finalState_eq,
    ← PiCCS.Transcript.deriveFromState_initialState,
    ProductionKey.key_oracle_eq, Transcript.piCcsOracle.initialState_is_prior]
  dsimp only [PiCCS.Transcript.deriveFromState]
  rw [deriveRoundsFrom_state, preRoundState_eq_run, ← run_append]

private theorem outgoingState_eq_run :
    ((ProductionKey.key relation ajtai).piCcsExecution running fresh proof).outgoingState =
      run Transcript.initialState (outputCalls fresh proof) := by
  rw [Nifs.PaperNonInteractive.Key.piCcsExecution_outgoingState_eq_absorbPiCcsOutput,
    ProductionKey.key_absorbPiCcsOutput, ProductionKey.absorbFullOutput, absorb_eq_run,
    roundsState_eq_run, ← run_append, outputCalls]

theorem rho_seal :
    (ProductionKey.key relation ajtai).piRlcChallenges running fresh proof =
      some fun index => readRho (run Transcript.initialState
        (outputCalls fresh proof ++ fixedCalls (.rho index))) := by
  unfold Nifs.PaperNonInteractive.Key.piRlcChallenges
  show ProductionKey.piRlcResponse
      ((ProductionKey.key relation ajtai).piCcsExecution running fresh proof).outgoingState = _
  rw [ProductionKey.piRlcResponse, Transcript.PiRlcSampler.piRlcChallenges,
    Transcript.PiRlcSampler.piRlcChallengesWithState_challenges, outgoingState_eq_run]
  refine congrArg some (funext fun index => ?_)
  rw [fixedCalls, run_append, run_map_widen]
  have read := Spec.Folding.Nifs.NonInteractive.PiRlcSampler.TranscriptHistory.queryAt_answer
    (run Transcript.initialState (outputCalls fresh proof)) [] (rhoIndex index)
  rewrite [Transcript.PiRlcSampler.sampleRingChallenge,
    Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.challengeAt,
    Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.scalarAt, readRho,
    ← Spec.Folding.Nifs.NonInteractive.PiRlcSampler.TranscriptHistory.answer, read]
  simp only [Spec.Folding.Nifs.NonInteractive.PiRlcSampler.TranscriptHistory.replay,
    List.foldl_nil, rhoIndex]

theorem challenge_seal (challenge : Challenge) :
    keyChallenge relation ajtai running fresh proof challenge =
      some (challenge.read (run Transcript.initialState
        (proverCalls fresh proof challenge ++ fixedCalls challenge))) := by
  cases challenge with
  | alpha coordinate =>
      exact congrArg some ((List.getD_eq_getElem _ K.zero _).symm.trans
        (alpha_seal relation ajtai running fresh proof coordinate))
  | gamma => exact congrArg some (gamma_seal relation ajtai running fresh proof)
  | round index =>
      exact congrArg some ((List.getD_eq_getElem _ K.zero _).symm.trans
        (round_seal relation ajtai running fresh proof index))
  | rho index =>
      simp only [keyChallenge, rho_seal relation ajtai running fresh proof]
      rfl

theorem coins_eq_reads :
    ((ProductionKey.key relation ajtai).piCcsExecution running fresh proof).coins =
      { alpha := ⟨List.ofFn fun coordinate => readK (slotOf coordinate.val)
            (run Transcript.initialState
              (proverCalls fresh proof (.alpha coordinate) ++ fixedCalls (.alpha coordinate))),
          by simp⟩
        gamma := readK (slotOf productionShape.cubeVariables) (run Transcript.initialState
          (proverCalls fresh proof .gamma ++ fixedCalls .gamma))
        roundPoint := ⟨List.ofFn fun index => readK (slotOf 0) (run Transcript.initialState
            (proverCalls fresh proof (.round index) ++ fixedCalls (.round index))),
          by simp⟩
        finalState := run Transcript.initialState (preRoundCalls fresh ++
          (canonicalFinIndices productionShape.cubeVariables).flatMap
            (messageCalls (messages proof))) } := by
  have alphaSeal := alpha_seal relation ajtai running fresh proof
  have pointSeal := round_seal relation ajtai running fresh proof
  have gammaSeal := gamma_seal relation ajtai running fresh proof
  have finalSeal := roundsState_eq_run relation ajtai running fresh proof
  revert alphaSeal pointSeal gammaSeal finalSeal
  generalize ((ProductionKey.key relation ajtai).piCcsExecution running fresh proof).coins =
    coins
  rcases coins with ⟨⟨alpha, alphaLength⟩, gamma, ⟨point, pointLength⟩, finalState⟩
  intro alphaSeal pointSeal gammaSeal finalSeal
  simp only at alphaSeal pointSeal gammaSeal finalSeal
  have alphaEqual : alpha = List.ofFn fun coordinate => readK (slotOf coordinate.val)
      (run Transcript.initialState
        (proverCalls fresh proof (.alpha coordinate) ++ fixedCalls (.alpha coordinate))) :=
    List.ext_getElem (by simp [alphaLength]) fun index inside _ =>
      (List.getD_eq_getElem _ K.zero inside).symm.trans
        ((alphaSeal ⟨index, alphaLength ▸ inside⟩).trans (by simp [proverCalls]))
  have pointEqual : point = List.ofFn fun index => readK (slotOf 0) (run Transcript.initialState
      (proverCalls fresh proof (.round index) ++ fixedCalls (.round index))) :=
    List.ext_getElem (by simp [pointLength]) fun index inside _ =>
      (List.getD_eq_getElem _ K.zero inside).symm.trans
        ((pointSeal ⟨index, pointLength ▸ inside⟩).trans (by simp [proverCalls]))
  subst alphaEqual pointEqual gammaSeal finalSeal
  rfl

theorem piCcsProbe_coins :
    ((ProductionKey.key relation ajtai).piCcsProbe running fresh proof).coins =
      coinsFrom (fun address => readK address.2 (run Transcript.initialState address.1))
        fresh proof := by
  simp only [Nifs.PaperNonInteractive.Key.piCcsProbe, coins_eq_reads relation ajtai running fresh proof]
  rfl

end Seals

private theorem messageCalls_length {degree : Nat} (left right : Proof degree)
    (round other : Fin productionShape.cubeVariables) :
    (messageCalls (messages left) round).length =
      (messageCalls (messages right) other).length := by
  simp [messageCalls, messages, Transcript.serializeMessage,
    SumCheck.Finite.FixedPolynomial.toMessage, (left.piCcsRounds round).coefficients_length,
    (right.piCcsRounds other).coefficients_length]

private theorem messageCalls_ne_nil {degree : Nat} (proof : Proof degree)
    (round : Fin productionShape.cubeVariables) :
    messageCalls (messages proof) round ≠ [] := by
  apply absorbCalls_ne_nil
  simp only [Transcript.serializeMessage, messages, SumCheck.Finite.FixedPolynomial.toMessage,
    ne_eq, List.flatMap_eq_nil_iff, not_forall]
  obtain ⟨coefficient, member⟩ := List.exists_mem_of_length_pos
    (l := (proof.piCcsRounds round).coefficients)
    (by rw [(proof.piCcsRounds round).coefficients_length]; omega)
  exact ⟨coefficient, member, by simp [serializeK]⟩

private theorem messageCalls_injective {degree : Nat} {left right : Proof degree}
    {round : Fin productionShape.cubeVariables}
    (same : messageCalls (messages left) round = messageCalls (messages right) round) :
    left.piCcsRounds round = right.piCcsRounds round :=
  SumCheck.Finite.FixedPolynomial.eq_of_coefficients (serializeKs_injective
    (absorbCalls_injective (by
      simp [(left.piCcsRounds round).coefficients_length,
        (right.piCcsRounds round).coefficients_length]) same))

private theorem messagesCalls_length {degree : Nat} (left right : Proof degree)
    (indices : List (Fin productionShape.cubeVariables)) :
    (indices.flatMap (messageCalls (messages left))).length =
      (indices.flatMap (messageCalls (messages right))).length :=
  flatMap_length_eq _ _ _ fun round _ => messageCalls_length left right round round

private theorem fullOutputWords_injective
    {left right : FullOutputCoordinates.FullOutput K productionShape}
    (same : ProductionKey.fullOutputWords left = ProductionKey.fullOutputWords right) :
    left = right := by
  have sources := fun source => serializeEvaluations_injective
    (flatMap_eq_of_lengths _ _ _ (fun _ _ => by simp) same source (List.mem_finRange source))
  have pad : left.padCoordinate = right.padCoordinate := funext fun source =>
    congrArg StrongReduction.EvaluationFamily.pad (sources source)
  have matrix : left.matrixCoordinate = right.matrixCoordinate := funext fun source =>
    congrArg StrongReduction.EvaluationFamily.matrix (sources source)
  cases left
  cases right
  simp only at pad matrix
  rw [pad, matrix]

private theorem statementWords_eq
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (ProductionKey.publicInputBlocks fresh).flatten =
      ProductionKey.priorDigest fresh ++
        (List.finRange productionShape.freshCount).flatMap fun index =>
          serializeCommitment (fresh.commitments index) ++
            serializePublicInput (fresh.publicInputs index) := by
  simp [ProductionKey.publicInputBlocks, List.flatMap, List.flatten_flatten, List.map_map,
    Function.comp_def]

private theorem statementWords_length
    (fresh fresh' : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (ProductionKey.publicInputBlocks fresh).flatten.length =
      (ProductionKey.publicInputBlocks fresh').flatten.length := by
  rw [statementWords_eq, statementWords_eq, List.length_append, List.length_append,
    flatMap_length_eq _ _
      (fun index => serializeCommitment (fresh'.commitments index) ++
        serializePublicInput (fresh'.publicInputs index))
      (fun _ _ => by simp)]
  simp [ProductionKey.priorDigest, decodeHash]

theorem statementCalls_length
    (fresh fresh' : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (statementCalls fresh).length = (statementCalls fresh').length := by
  simp only [statementCalls, List.length_append, absorbCalls_length,
    statementWords_length fresh fresh']

theorem statementCalls_identify_fresh
    {fresh fresh' : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)}
    (same : statementCalls fresh = statementCalls fresh') : fresh = fresh' := by
  have words := absorbCalls_injective (statementWords_length fresh fresh')
    (List.append_cancel_left same)
  rw [statementWords_eq, statementWords_eq] at words
  obtain ⟨_, perFresh⟩ := List.append_inj words
    (by simp [ProductionKey.priorDigest, decodeHash])
  have each := flatMap_eq_of_lengths _ _ _ (fun _ _ => by simp) perFresh
  have parts := fun index => List.append_inj (each index (List.mem_finRange index))
    (by simp)
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

private theorem preRoundCalls_length
    (fresh fresh' : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (preRoundCalls fresh).length = (preRoundCalls fresh').length := by
  simp only [preRoundCalls, List.length_append, statementCalls_length fresh fresh']

private theorem preRoundCalls_identify
    {fresh fresh' : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)}
    {rest rest' : List Call}
    (same : preRoundCalls fresh ++ rest = preRoundCalls fresh' ++ rest') :
    fresh = fresh' ∧ rest = rest' := by
  obtain ⟨preRoundEqual, restEqual⟩ := List.append_inj same (preRoundCalls_length fresh fresh')
  exact ⟨statementCalls_identify_fresh
    (List.append_inj preRoundEqual (statementCalls_length fresh fresh')).1, restEqual⟩

private theorem roundPrefixCalls_identify {degree : Nat}
    {fresh fresh' : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)}
    {proof proof' : Proof degree} {round : Fin productionShape.cubeVariables}
    (same : roundPrefixCalls fresh proof round = roundPrefixCalls fresh' proof' round) :
    fresh = fresh' ∧ ∀ earlier : Fin productionShape.cubeVariables,
      earlier.val ≤ round.val → proof.piCcsRounds earlier = proof'.piCcsRounds earlier := by
  simp only [roundPrefixCalls, List.append_assoc] at same
  obtain ⟨freshEqual, afterPreRound⟩ := preRoundCalls_identify same
  obtain ⟨earlierEqual, ownEqual⟩ :=
    List.append_inj afterPreRound (messagesCalls_length proof proof' _)
  refine ⟨freshEqual, fun earlier atMost => ?_⟩
  rcases Nat.lt_or_eq_of_le atMost with before | equal
  · exact messageCalls_injective (flatMap_eq_of_lengths _ _ _
      (fun round _ => messageCalls_length proof proof' round round) earlierEqual earlier
      (mem_take_canonicalFinIndices earlier before))
  · rw [Fin.ext equal]
    exact messageCalls_injective ownEqual

private theorem outputCalls_identify {degree : Nat}
    {fresh fresh' : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)}
    {proof proof' : Proof degree}
    (same : outputCalls fresh proof = outputCalls fresh' proof') :
    fresh = fresh' ∧
      { proof with
        piDecCommitments := proof'.piDecCommitments
        piDecEvaluations := proof'.piDecEvaluations } = proof' := by
  simp only [outputCalls, List.append_assoc] at same
  obtain ⟨freshEqual, afterPreRound⟩ := preRoundCalls_identify same
  obtain ⟨roundsEqual, outputEqual⟩ :=
    List.append_inj afterPreRound (messagesCalls_length proof proof' _)
  have rounds : proof.piCcsRounds = proof'.piCcsRounds := funext fun round =>
    messageCalls_injective (flatMap_eq_of_lengths _ _ _
      (fun round _ => messageCalls_length proof proof' round round) roundsEqual round
      (mem_canonicalFinIndices round))
  have output : proof.piCcsOutput = proof'.piCcsOutput :=
    fullOutputWords_injective (absorbCalls_injective
      (by simp [ProductionKey.fullOutputWords, List.length_flatMap]) outputEqual)
  refine ⟨freshEqual, ?_⟩
  cases proof
  cases proof'
  simp only at rounds output
  subst rounds output
  rfl

private theorem cubeVariables_eq : productionShape.cubeVariables = 28 := rfl

private theorem take_messages_length {degree : Nat} (proof : Proof degree)
    (round : Fin productionShape.cubeVariables) :
    (((canonicalFinIndices productionShape.cubeVariables).take round.val).flatMap
      (messageCalls (messages proof))).length =
      round.val * (messageCalls (messages proof) ⟨0, by decide⟩).length := by
  rw [flatMap_length_constant _ _ ((messageCalls (messages proof) ⟨0, by decide⟩).length)
    fun other => messageCalls_length proof proof other ⟨0, by decide⟩]
  simp [canonicalFinIndices, cubeVariables_eq]
  exact Or.inl (Nat.le_of_lt round.isLt)

private def offset (count output : Nat) : Challenge → Nat
  | .alpha coordinate => coordinate.val / 6
  | .gamma => 4
  | .round index => 4 + index.val * count + count
  | .rho index => 4 + 28 * count + output + 2 * index.val + 1

private theorem offset_slot_injective (count output : Nat) (positive : 0 < count)
    {challenge other : Challenge}
    (same : offset count output challenge = offset count output other)
    (sameSlot : challenge.slot = other.slot) : challenge = other := by
  have product (round : Fin productionShape.cubeVariables) :
      round.val * count ≤ 27 * count :=
    Nat.mul_le_mul_right _ (by have below : round.val < 28 := round.isLt; omega)
  have slotValue := congrArg Fin.val sameSlot
  match challenge, other with
  | .alpha coordinate, .alpha coordinate' =>
      simp only [offset, Challenge.slot, slotOf, Transcript.coinsPerChunk_eq] at same slotValue
      exact congrArg Challenge.alpha (Fin.ext (by omega))
  | .alpha coordinate, .gamma =>
      have below := coordinate.isLt
      simp only [offset, Challenge.slot, slotOf, Transcript.coinsPerChunk_eq,
        cubeVariables_eq] at same slotValue below
      omega
  | .alpha coordinate, .round index =>
      have below := coordinate.isLt
      simp only [offset, cubeVariables_eq] at same below
      generalize index.val * count = steps at same
      omega
  | .alpha coordinate, .rho _ =>
      have below := coordinate.isLt
      simp only [offset, cubeVariables_eq] at same below
      omega
  | .gamma, .alpha coordinate =>
      have below := coordinate.isLt
      simp only [offset, Challenge.slot, slotOf, Transcript.coinsPerChunk_eq,
        cubeVariables_eq] at same slotValue below
      omega
  | .gamma, .gamma => rfl
  | .gamma, .round index =>
      simp only [offset] at same
      generalize index.val * count = steps at same
      omega
  | .gamma, .rho _ => simp only [offset] at same; omega
  | .round index, .alpha coordinate =>
      have below := coordinate.isLt
      simp only [offset, cubeVariables_eq] at same below
      generalize index.val * count = steps at same
      omega
  | .round index, .gamma =>
      simp only [offset] at same
      generalize index.val * count = steps at same
      omega
  | .round index, .round index' =>
      simp only [offset] at same
      have steps : index.val * count = index'.val * count := by omega
      exact congrArg Challenge.round (Fin.ext (Nat.eq_of_mul_eq_mul_right positive steps))
  | .round index, .rho _ =>
      simp only [offset] at same
      have bound := product index
      generalize index.val * count = steps at same bound
      omega
  | .rho _, .alpha coordinate =>
      have below := coordinate.isLt
      simp only [offset, cubeVariables_eq] at same below
      omega
  | .rho _, .gamma => simp only [offset] at same; omega
  | .rho _, .round index' =>
      simp only [offset] at same
      have bound := product index'
      generalize index'.val * count = steps at same bound
      omega
  | .rho scalar, .rho scalar' =>
      simp only [offset] at same
      exact congrArg Challenge.rho (Fin.ext (by omega))

private theorem challengeCalls_length {degree : Nat}
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof degree) (challenge : Challenge) :
    (challengeCalls fresh proof challenge).length = (statementCalls fresh).length +
      offset (messageCalls (messages proof) ⟨0, by decide⟩).length
        (absorbCalls (ProductionKey.fullOutputWords proof.piCcsOutput)).length
        challenge := by
  cases challenge with
  | alpha coordinate =>
      simp only [challengeCalls, proverCalls, fixedCalls, List.length_append,
        List.length_replicate, offset, Transcript.coinsPerChunk_eq]
  | gamma =>
      simp only [challengeCalls, proverCalls, fixedCalls, List.length_append,
        List.length_replicate, offset, Transcript.coinsPerChunk_eq, cubeVariables_eq]
  | round index =>
      simp only [challengeCalls, proverCalls, fixedCalls, roundPrefixCalls, preRoundCalls,
        List.length_append]
      rw [take_messages_length, messageCalls_length proof proof index ⟨0, by decide⟩]
      simp only [List.length_replicate, List.length_nil, offset, Transcript.coinsPerChunk_eq,
        cubeVariables_eq]
      omega
  | rho index =>
      have rounds : ((canonicalFinIndices productionShape.cubeVariables).flatMap
          (messageCalls (messages proof))).length =
          28 * (messageCalls (messages proof) ⟨0, by decide⟩).length := by
        rw [flatMap_length_constant _ _ ((messageCalls (messages proof) ⟨0, by decide⟩).length)
          fun other => messageCalls_length proof proof other ⟨0, by decide⟩]
        simp [canonicalFinIndices, cubeVariables_eq]
      have scalars : ((Spec.Folding.Nifs.NonInteractive.PiRlcSampler.ScheduleLaw.queryAt []
          (rhoIndex index)).val.map widen).length = 2 * index.val + 1 := by
        rw [List.length_map, Spec.Folding.Nifs.NonInteractive.PiRlcSampler.ScheduleLaw.queryAt_length]
        simp [rhoIndex]
      simp only [challengeCalls, proverCalls, fixedCalls, outputCalls, preRoundCalls,
        List.length_append]
      rw [rounds, scalars]
      simp only [List.length_replicate, offset, Transcript.coinsPerChunk_eq, cubeVariables_eq]
      omega

theorem challengeCalls_length_eq {degree : Nat}
    (fresh fresh' : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof proof' : Proof degree) (challenge : Challenge) :
    (challengeCalls fresh proof challenge).length =
      (challengeCalls fresh' proof' challenge).length := by
  have outputLength :
      (absorbCalls (ProductionKey.fullOutputWords proof.piCcsOutput)).length =
        (absorbCalls (ProductionKey.fullOutputWords proof'.piCcsOutput)).length := by
    simp [ProductionKey.fullOutputWords, List.length_flatMap]
  rw [challengeCalls_length, challengeCalls_length, statementCalls_length fresh fresh',
    messageCalls_length proof proof' ⟨0, by decide⟩ ⟨0, by decide⟩, outputLength]

theorem challengeAddress_injective {degree : Nat}
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof degree) {challenge other : Challenge}
    (same : challengeAddress fresh proof challenge = challengeAddress fresh proof other) :
    challenge = other := by
  have lengths := congrArg (fun address => address.1.length) same
  simp only [challengeAddress, challengeCalls_length] at lengths
  exact offset_slot_injective _ _
    (List.length_pos_iff.mpr (messageCalls_ne_nil proof ⟨0, by decide⟩))
    (Nat.add_left_cancel lengths) (congrArg Prod.snd same)

private theorem absorbCalls_head? {words : List F} (nonempty : words ≠ []) :
    (absorbCalls words).head? = some (padChunk (words.take Poseidon2.rate)) := by
  have positive : 0 < (words.length + Poseidon2.rate - 1) / Poseidon2.rate := by
    have length : 0 < words.length := List.length_pos_iff.mpr nonempty
    unfold Poseidon2.rate
    omega
  obtain ⟨count, count_eq⟩ : ∃ count, (words.length + Poseidon2.rate - 1) / Poseidon2.rate =
      count + 1 := ⟨_, (Nat.succ_pred_eq_of_pos positive).symm⟩
  unfold absorbCalls
  rw [count_eq, List.range_succ_eq_map, List.map_cons, List.head?_cons, Nat.zero_mul, List.drop_zero]

private theorem challengeCalls_head? {degree : Nat}
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof degree) (challenge : Challenge) :
    (challengeCalls fresh proof challenge).head? = some (padChunk Transcript.foldDomainChunk) := by
  have statement : (statementCalls fresh).head? = some (padChunk Transcript.foldDomainChunk) := by
    unfold statementCalls
    rw [List.head?_append, absorbCalls_head? (by simp [Transcript.foldDomainChunk, Poseidon2.rate]),
      List.take_of_length_le (by simp [Poseidon2.rate])]
    rfl
  cases challenge <;>
    simp only [challengeCalls, proverCalls, roundPrefixCalls, outputCalls, preRoundCalls,
      List.append_assoc, List.head?_append, statement, Option.some_or]

variable (logicalWidth publicFits) in
def stateHashCalls (preimage : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    List Call :=
  absorbCalls (serializePreimage (publicFits := publicFits) preimage) ++ [padChunk [1]]

theorem stateHash_eq_run
    (preimage : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    stateHash (publicFits := publicFits) preimage =
      (run Transcript.initialState (stateHashCalls logicalWidth publicFits preimage)).take
        Poseidon2.digestLen := by
  have padded (words : List F) : Poseidon2.hash words =
      (Poseidon2.absorbBlock (Transcript.absorb Transcript.initialState words) [1]).take
        Poseidon2.digestLen := by
    unfold Poseidon2.hash Transcript.absorb Transcript.initialState
    dsimp only
    rw [Poseidon2.absorbBlock]
    refine congrArg (fun state => (Poseidon2.permute state).take Poseidon2.digestLen) ?_
    refine List.map_congr_left fun lane _ => ?_
    cases lane with
    | zero => simp
    | succ lane => simp
  unfold stateHash stateHashCalls
  rw [padded, run_append, ← absorb_eq_run, run_cons, run_nil, step,
    ← absorbBlock_padChunk _ (by simp [Poseidon2.rate])]

theorem stateHash_separated {degree : Nat}
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof degree) (challenge : Challenge)
    (preimage : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits))
    {calls : List Call} (challengePrefix : calls <+: challengeCalls fresh proof challenge)
    (hashPrefix : calls <+: stateHashCalls logicalWidth publicFits preimage) :
    calls = [] := by
  cases calls with
  | nil => rfl
  | cons call rest =>
      exfalso
      have hashHead : (stateHashCalls logicalWidth publicFits preimage).head? =
          some (padChunk stateDomainChunk) := by
        unfold stateHashCalls serializePreimage
        have rate : stateDomainChunk.length = Poseidon2.rate := stateDomainChunk_length
        rw [List.head?_append, absorbCalls_head? (by simp [stateDomainChunk, Poseidon2.rate]),
          List.append_assoc, List.take_left' rate]
        rfl
      obtain ⟨challengeTail, challengeEq⟩ := challengePrefix
      obtain ⟨hashTail, hashEq⟩ := hashPrefix
      have fromChallenge := challengeCalls_head? fresh proof challenge
      rw [← challengeEq] at fromChallenge
      rw [← hashEq] at hashHead
      have same : padChunk Transcript.foldDomainChunk = padChunk stateDomainChunk :=
        Option.some_injective _ (fromChallenge.symm.trans hashHead)
      have lane := congrFun same ⟨0, by decide⟩
      revert lane
      decide

/-- Normative: a change to this definition is a protocol change, pinned by
`tests/TranscriptCoverageChecks.lean`. SuperNeo v1.2 Sections 7.3–7.4 order. -/
def AgreeOnAbsorbed {degree : Nat}
    (fresh fresh' : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof proof' : Proof degree) : Challenge → Prop
  | .alpha _ => fresh = fresh'
  | .gamma => fresh = fresh'
  | .round index => fresh = fresh' ∧ ∀ earlier : Fin productionShape.cubeVariables,
      earlier.val ≤ index.val → proof.piCcsRounds earlier = proof'.piCcsRounds earlier
  | .rho _ => fresh = fresh' ∧
      { proof with
        piDecCommitments := proof'.piDecCommitments
        piDecEvaluations := proof'.piDecEvaluations } = proof'

theorem AgreeOnAbsorbed.fresh_eq {degree : Nat}
    {fresh fresh' : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)}
    {proof proof' : Proof degree} {challenge : Challenge}
    (agree : AgreeOnAbsorbed fresh fresh' proof proof' challenge) : fresh = fresh' := by
  cases challenge with
  | alpha => exact agree
  | gamma => exact agree
  | round => exact agree.1
  | rho => exact agree.1

theorem proverCalls_identify {degree : Nat}
    {fresh fresh' : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)}
    {proof proof' : Proof degree} (challenge : Challenge)
    (same : proverCalls fresh proof challenge = proverCalls fresh' proof' challenge) :
    AgreeOnAbsorbed fresh fresh' proof proof' challenge := by
  cases challenge with
  | alpha => exact statementCalls_identify_fresh same
  | gamma => exact statementCalls_identify_fresh same
  | round => exact roundPrefixCalls_identify same
  | rho => exact outputCalls_identify same

end Contract

end NightstreamFPrime.Lifecycle.TranscriptCoverage

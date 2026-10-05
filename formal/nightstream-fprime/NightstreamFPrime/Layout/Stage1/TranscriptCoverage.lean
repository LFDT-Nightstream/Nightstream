import NightstreamFPrime.Layout.Stage1.PiCCSSecurity
import NightstreamFPrime.Lifecycle.ProductionKey

/-!
Owns the Fiat–Shamir coverage contract of the production NIFS transcript.

Every verifier challenge (`α`, `γ`, the SumCheck points, and the `Π_RLC`
scalars) is a read of the Poseidon2 state that an explicit list of duplex
calls reaches. A duplex call is a zero-padded rate chunk or a squeeze, as the
permutation sees it.

Inputs: the production key, the running and fresh statements, the NIFS proof,
and the prior hash preimage.

Outputs:
- seal theorems: each key challenge equals the read of its call list;
- coverage theorems: equal prover-dependent calls before a challenge give
  equal fresh statements and equal earlier prover messages;
- `calls_identify_view_or_collision`: with the prior-state link, equal calls
  also identify the running statement and the verifier context, unless the
  state hash collides.

Invariant: removing an absorption of verifier-relevant data, moving it after
a challenge that depends on it, or making its encoding ambiguous breaks a seal
or a coverage proof.

Does not own: Poseidon2 security, the Fiat–Shamir transfer (an approved
external assumption), or the circuit refinement of this schedule. The `Π_DEC`
child messages follow the last challenge; the next state hash binds them.
-/

namespace NightstreamFPrime.Layout.Stage1.TranscriptCoverage

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

/-! ## Duplex calls -/

/-- One Poseidon2 duplex call as the permutation sees it. -/
inductive Call where
  | absorb (chunk : Fin Poseidon2.rate → F)
  | squeeze

/-- Apply one call to the sponge state. -/
def step (state : Transcript.State) : Call → Transcript.State
  | .absorb chunk => Poseidon2.absorbBlock state (List.ofFn chunk)
  | .squeeze => Poseidon2.permute state

/-- Apply a call list in order. -/
def run (state : Transcript.State) (calls : List Call) : Transcript.State :=
  calls.foldl step state

@[simp] private theorem run_nil (state : Transcript.State) : run state [] = state := rfl

@[simp] private theorem run_cons (state : Transcript.State) (call : Call) (calls : List Call) :
    run state (call :: calls) = run (step state call) calls := rfl

@[simp] private theorem step_squeeze (state : Transcript.State) :
    step state .squeeze = Poseidon2.permute state := rfl

private theorem run_append (state : Transcript.State) (left right : List Call) :
    run state (left ++ right) = run (run state left) right := by
  simp [run, List.foldl_append]

/-- The rate chunk that the sponge adds for a short word list. -/
def padChunk (words : List F) : Fin Poseidon2.rate → F :=
  fun lane => words.getD lane.val 0

/-- Additive absorption reads missing chunk words as zero. -/
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

/-- The calls of `Transcript.absorb`: one padded chunk per started rate block. -/
def absorbCalls (words : List F) : List Call :=
  (List.range ((words.length + Poseidon2.rate - 1) / Poseidon2.rate)).map fun chunk =>
    .absorb (padChunk ((words.drop (chunk * Poseidon2.rate)).take Poseidon2.rate))

theorem absorb_eq_run (state : Transcript.State) (words : List F) :
    Transcript.absorb state words = run state (absorbCalls words) := by
  unfold Transcript.absorb absorbCalls run
  rw [List.foldl_map, List.foldl_map]
  apply List.foldl_ext
  intro current chunk _
  exact absorbBlock_padChunk current (List.length_take_le _ _)

@[simp] private theorem absorbCalls_length (words : List F) :
    (absorbCalls words).length = (words.length + Poseidon2.rate - 1) / Poseidon2.rate := by
  simp [absorbCalls]

/-- Equal-length word lists with equal padded chunks are equal: zero padding
creates no ambiguity when the length is fixed. -/
theorem absorbCalls_injective {left right : List F}
    (lengthEqual : left.length = right.length)
    (same : absorbCalls left = absorbCalls right) : left = right := by
  have chunks : ∀ chunk ∈ List.range ((left.length + Poseidon2.rate - 1) / Poseidon2.rate),
      Call.absorb (padChunk ((left.drop (chunk * Poseidon2.rate)).take Poseidon2.rate)) =
        Call.absorb (padChunk ((right.drop (chunk * Poseidon2.rate)).take Poseidon2.rate)) := by
    unfold absorbCalls at same
    rw [← lengthEqual] at same
    exact List.map_inj_left.mp same
  apply List.ext_getElem lengthEqual
  intro position leftBound rightBound
  have chunkBound : position / Poseidon2.rate <
      (left.length + Poseidon2.rate - 1) / Poseidon2.rate := by
    unfold Poseidon2.rate at *
    omega
  have entry := Call.absorb.inj
    (chunks (position / Poseidon2.rate) (List.mem_range.mpr chunkBound))
  have lane := congrFun entry
    ⟨position % Poseidon2.rate, Nat.mod_lt _ (by decide)⟩
  have split : position / Poseidon2.rate * Poseidon2.rate +
      position % Poseidon2.rate = position := by
    rw [Nat.mul_comm]
    exact Nat.div_add_mod position Poseidon2.rate
  simp only [padChunk, List.getD_eq_getElem?_getD, List.getElem?_take,
    Nat.mod_lt _ (show 0 < Poseidon2.rate by decide), if_true,
    List.getElem?_drop, split] at lane
  simpa [List.getElem?_eq_getElem leftBound, List.getElem?_eq_getElem rightBound]
    using lane

/-- The extension challenge that `Transcript.squeezeK` reads before its two
squeezes. -/
def readK (state : Transcript.State) : K :=
  ⟨state.getD 0 0, (Poseidon2.permute state).getD 0 0⟩

private theorem squeezeK_eq (state : Transcript.State) :
    Transcript.squeezeK state = (readK state, run state [.squeeze, .squeeze]) := by
  simp only [Transcript.squeezeK, Transcript.squeezeF, readK, run_cons, run_nil, step]

/-- One labelled PiCCS squeeze: absorb the label words, then squeeze twice. -/
def labelCalls (label : FiatShamir.ChallengeLabel productionShape) : List Call :=
  absorbCalls (Transcript.labelWord label) ++ [.squeeze, .squeeze]

private theorem oracle_squeeze_eq (state : Transcript.State)
    (label : FiatShamir.ChallengeLabel productionShape) :
    Transcript.piCcsOracle.transcript.squeeze state label =
      (readK (run state (absorbCalls (Transcript.labelWord label))),
        run state (labelCalls label)) := by
  change Transcript.squeezeK (Transcript.absorb state (Transcript.labelWord label)) = _
  rw [squeezeK_eq, absorb_eq_run, labelCalls, run_append]

/-- The calls of `Transcript.absorbBlock`: the length-prefixed block. -/
private theorem absorbBlock_eq_run (state : Transcript.State) (words : List F) :
    Transcript.absorbBlock state words = run state (absorbCalls (block words)) :=
  absorb_eq_run state (block words)

private theorem absorbBlocks_eq_run (state : Transcript.State) (blocks : List (List F)) :
    Transcript.absorbBlocks state blocks =
      run state (blocks.flatMap fun words => absorbCalls (block words)) := by
  induction blocks generalizing state with
  | nil => rfl
  | cons words blocks inductionHypothesis =>
      simp only [Transcript.absorbBlocks, List.foldl_cons, List.flatMap_cons, run_append]
      rw [← absorbBlock_eq_run]
      exact inductionHypothesis _

/-! ## Labelled squeezes and SumCheck rounds -/

/-- The pre-SumCheck state: every `α` squeeze, then the `γ` squeeze. Stated
at an abstract shape so that no concrete round list is unfolded. -/
private theorem derivePreSumcheck_state {Context Field State : Type*} {shape : Shape}
    (oracle : FiatShamir.Oracle Context Field State shape) (context : Context) :
    (FiatShamir.derivePreSumcheck oracle context).state =
      (oracle.squeeze (FiatShamir.squeezeMany oracle (oracle.initialState context)
        (FiatShamir.alphaLabels shape)).2 .gamma).2 := rfl

/-- The final PiCCS state: every round after the pre-SumCheck state. -/
private theorem derive_finalState {Context Field State : Type*} {shape : Shape}
    (oracle : FiatShamir.Oracle Context Field State shape) (context : Context)
    (certificate : FiatShamir.Certificate Field shape) :
    (FiatShamir.derive oracle context certificate).finalState =
      (FiatShamir.deriveRoundsFrom oracle certificate.rounds
        (FiatShamir.derivePreSumcheck oracle context).state
        (canonicalFinIndices shape.cubeVariables)).2 := rfl

/-- The state after a list of labelled squeezes. -/
private theorem squeezeMany_state (state : Transcript.State)
    (labels : List (FiatShamir.ChallengeLabel productionShape)) :
    (FiatShamir.squeezeMany Transcript.piCcsOracle.transcript state labels).2 =
      run state (labels.flatMap labelCalls) := by
  induction labels generalizing state with
  | nil => rfl
  | cons label labels inductionHypothesis =>
      simp only [FiatShamir.squeezeMany, oracle_squeeze_eq, List.flatMap_cons, run_append]
      exact inductionHypothesis _

/-- Each labelled squeeze reads the state after every earlier labelled
squeeze and its own label words. -/
private theorem squeezeMany_getD (state : Transcript.State)
    (labels : List (FiatShamir.ChallengeLabel productionShape))
    (index : Nat) (bound : index < labels.length) :
    (FiatShamir.squeezeMany Transcript.piCcsOracle.transcript state labels).1.getD
        index K.zero =
      readK (run state ((labels.take index).flatMap labelCalls ++
        absorbCalls (Transcript.labelWord labels[index]))) := by
  induction labels generalizing state index with
  | nil => simp at bound
  | cons label labels inductionHypothesis =>
      cases index with
      | zero => simp [FiatShamir.squeezeMany, oracle_squeeze_eq]
      | succ index =>
          simp only [FiatShamir.squeezeMany, oracle_squeeze_eq, List.getD_cons_succ,
            List.take_succ_cons, List.flatMap_cons, List.getElem_cons_succ]
          rw [inductionHypothesis _ index (by simpa using bound), ← run_append,
            List.append_assoc]

/-- The canonical round point uses the identity enumeration. -/
private theorem canonicalFinIndices_getElem (count index : Nat)
    (bound : index < (canonicalFinIndices count).length) :
    (canonicalFinIndices count)[index] = ⟨index, by simpa [canonicalFinIndices] using bound⟩ := by
  simp [canonicalFinIndices]

/-- The SumCheck messages carried by one NIFS proof. -/
def messages {degree : Nat} (proof : Proof degree) :
    Fin productionShape.cubeVariables → SumCheck.Finite.Message K :=
  fun round => (proof.piCcsRounds round).toMessage

/-- The absorbed block of one indexed SumCheck message. -/
def messageCalls (rounds : Fin productionShape.cubeVariables → SumCheck.Finite.Message K)
    (round : Fin productionShape.cubeVariables) : List Call :=
  absorbCalls (block (natWord round.val :: Transcript.serializeMessage (rounds round)))

/-- One SumCheck round: absorb the message, then squeeze its challenge. -/
def roundCalls (rounds : Fin productionShape.cubeVariables → SumCheck.Finite.Message K)
    (round : Fin productionShape.cubeVariables) : List Call :=
  messageCalls rounds round ++ labelCalls (.sumcheck round)

private theorem oracle_absorbRound_eq (state : Transcript.State)
    (rounds : Fin productionShape.cubeVariables → SumCheck.Finite.Message K)
    (round : Fin productionShape.cubeVariables) :
    Transcript.piCcsOracle.transcript.absorbRound state round (rounds round) =
      run state (messageCalls rounds round) :=
  absorbBlock_eq_run state _

private theorem deriveRoundsFrom_state
    (rounds : Fin productionShape.cubeVariables → SumCheck.Finite.Message K)
    (state : Transcript.State) (indices : List (Fin productionShape.cubeVariables)) :
    (FiatShamir.deriveRoundsFrom Transcript.piCcsOracle.transcript rounds state indices).2 =
      run state (indices.flatMap (roundCalls rounds)) := by
  induction indices generalizing state with
  | nil => rfl
  | cons round indices inductionHypothesis =>
      simp only [FiatShamir.deriveRoundsFrom, oracle_squeeze_eq, List.flatMap_cons,
        roundCalls, run_append, oracle_absorbRound_eq]
      rw [inductionHypothesis]

private theorem deriveRoundsFrom_getD
    (rounds : Fin productionShape.cubeVariables → SumCheck.Finite.Message K)
    (state : Transcript.State) (indices : List (Fin productionShape.cubeVariables))
    (index : Nat) (bound : index < indices.length) :
    (FiatShamir.deriveRoundsFrom Transcript.piCcsOracle.transcript rounds state
        indices).1.getD index K.zero =
      readK (run state ((indices.take index).flatMap (roundCalls rounds) ++
        messageCalls rounds indices[index] ++
        absorbCalls (Transcript.labelWord (.sumcheck indices[index])))) := by
  induction indices generalizing state index with
  | nil => simp at bound
  | cons round indices inductionHypothesis =>
      cases index with
      | zero =>
          simp only [FiatShamir.deriveRoundsFrom, oracle_squeeze_eq, List.getD_cons_zero,
            List.take_zero, List.flatMap_nil, List.nil_append, List.getElem_cons_zero,
            run_append, oracle_absorbRound_eq]
      | succ index =>
          simp only [FiatShamir.deriveRoundsFrom, oracle_squeeze_eq, List.getD_cons_succ,
            List.take_succ_cons, List.flatMap_cons, List.getElem_cons_succ]
          rw [inductionHypothesis _ index (by simpa using bound)]
          simp only [roundCalls, run_append, List.append_assoc, oracle_absorbRound_eq]

/-! ## Production schedule and seals -/

section Schedule

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- Calls before the first `α` label: the digest-only domain tag and the key's
public-input blocks. -/
def statementCalls
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)) : List Call :=
  absorbCalls Transcript.piCcsDigestDomainTag ++
    (ProductionKey.publicInputBlocks running fresh).flatMap fun words =>
      absorbCalls (block words)

/-- Calls before the first SumCheck message: the statement, then every
labelled `α` squeeze and the `γ` squeeze. -/
def preRoundCalls
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)) : List Call :=
  statementCalls running fresh ++
    (FiatShamir.alphaLabels productionShape).flatMap labelCalls ++ labelCalls .gamma

/-- Prover-dependent calls before the round `round` challenge: every earlier
round and the round's own message. Its label words follow. -/
def roundPrefixCalls {degree : Nat}
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof degree) (round : Fin productionShape.cubeVariables) : List Call :=
  preRoundCalls running fresh ++
    ((canonicalFinIndices productionShape.cubeVariables).take round.val).flatMap
      (roundCalls (messages proof)) ++
    messageCalls (messages proof) round

/-- The `y′` words, in the key's canonical source/Pad/matrix order. -/
def outputWords (output : FullOutputCoordinates.FullOutput K productionShape) : List F :=
  (List.finRange productionShape.sourceCount).flatMap fun source =>
    ((List.finRange productionShape.coefficientCount).flatMap fun coefficient =>
      serializeK (output.padCoordinate source coefficient)) ++
    ((List.finRange productionShape.matrixCount).flatMap fun matrix =>
      (List.finRange productionShape.coefficientCount).flatMap fun coefficient =>
        serializeK (output.matrixCoordinate source matrix coefficient))

/-- Seal: the key absorbs exactly `outputWords` as one block. -/
private theorem absorbFullOutput_eq_run (state : Transcript.State)
    (output : FullOutputCoordinates.FullOutput K productionShape) :
    ProductionKey.absorbFullOutput state output =
      run state (absorbCalls (block (outputWords output))) :=
  absorbBlock_eq_run state (outputWords output)

/-- Calls before the first `Π_RLC` scalar: every round and the complete `y′`. -/
def outputCalls {degree : Nat}
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof degree) : List Call :=
  preRoundCalls running fresh ++
    (canonicalFinIndices productionShape.cubeVariables).flatMap (roundCalls (messages proof)) ++
    absorbCalls (block (outputWords proof.piCcsOutput))

/-- The `Π_RLC` domain chunk `[4, coordinate]`. -/
def rhoEnterCall (coordinate : Nat) : Call :=
  .absorb (padChunk [Poseidon2.ofNat 4, Poseidon2.ofNat coordinate])

/-- Fixed calls between the complete `y′` and the read of one `Π_RLC` scalar. -/
def rhoReadCalls (coordinate : Nat) : List Call :=
  ((List.range coordinate).flatMap fun earlier => [rhoEnterCall earlier, .squeeze]) ++
    [rhoEnterCall coordinate]

/-- The `Π_RLC` challenge that the sampler reads from four state lanes. -/
def readRho (state : Transcript.State) : RingF :=
  Phi81StrongSet.embedScalar
    (Spec.Folding.Nifs.NonInteractive.PiRlcSampler.sample
      (Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.block state))

private theorem rhoEnter_eq_step (state : Transcript.State) (coordinate : Nat) :
    Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.enter state coordinate =
      step state (rhoEnterCall coordinate) :=
  absorbBlock_padChunk state (by simp [Poseidon2.rate])

private theorem rhoStateAt_eq_run (state : Transcript.State) (coordinate : Nat) :
    Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.stateAt state coordinate =
      run state ((List.range coordinate).flatMap fun earlier =>
        [rhoEnterCall earlier, .squeeze]) := by
  induction coordinate with
  | zero => rfl
  | succ coordinate inductionHypothesis =>
      rw [Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.stateAt_succ,
        inductionHypothesis, List.range_succ, List.flatMap_append, run_append,
        rhoEnter_eq_step]
      simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, run_cons, run_nil,
        step_squeeze]

private theorem publicInputState_eq_run
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (ProductionKey.key relation ajtai).publicInputState running fresh =
      run Transcript.initialState (statementCalls running fresh) := by
  rw [ProductionKey.key_publicInputState_eq, absorbBlocks_eq_run, absorb_eq_run,
    statementCalls, run_append]

/-- Seal for `α`: coordinate `i` reads the state after the statement, the
earlier `α` squeezes, and its own label. -/
theorem alpha_seal
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof (ProductionKey.degreeBound relation))
    (coordinate : Fin productionShape.cubeVariables) :
    ((ProductionKey.key relation ajtai).piCcsExecution running fresh
        proof).coins.alpha.coordinates.getD coordinate.val K.zero =
      readK (run Transcript.initialState (statementCalls running fresh ++
        ((FiatShamir.alphaLabels productionShape).take coordinate.val).flatMap labelCalls ++
        absorbCalls (Transcript.labelWord (.alpha coordinate)))) := by
  rw [Nifs.PaperNonInteractive.Key.piCcsExecution_coins_eq_derive]
  change (FiatShamir.squeezeMany Transcript.piCcsOracle.transcript
      ((ProductionKey.key relation ajtai).publicInputState running fresh)
      (FiatShamir.alphaLabels productionShape)).1.getD coordinate.val K.zero = _
  rw [squeezeMany_getD _ _ _ (by simp [FiatShamir.alphaLabels_length]),
    publicInputState_eq_run, ← run_append, List.append_assoc]
  simp [FiatShamir.alphaLabels, canonicalFinIndices]

/-- Seal for `γ`: it reads the state after every `α` squeeze and its label. -/
theorem gamma_seal
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof (ProductionKey.degreeBound relation)) :
    ((ProductionKey.key relation ajtai).piCcsExecution running fresh proof).coins.gamma =
      readK (run Transcript.initialState (statementCalls running fresh ++
        (FiatShamir.alphaLabels productionShape).flatMap labelCalls ++
        absorbCalls (Transcript.labelWord .gamma))) := by
  rw [Nifs.PaperNonInteractive.Key.piCcsExecution_coins_eq_derive]
  change (Transcript.piCcsOracle.transcript.squeeze
      (FiatShamir.squeezeMany Transcript.piCcsOracle.transcript
        ((ProductionKey.key relation ajtai).publicInputState running fresh)
        (FiatShamir.alphaLabels productionShape)).2 .gamma).1 = _
  simp only [oracle_squeeze_eq, squeezeMany_state, publicInputState_eq_run, ← run_append,
    List.append_assoc]

/-- The state after `γ`, before the first SumCheck message. -/
private theorem preRoundState_eq_run
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (Transcript.piCcsOracle.transcript.squeeze
        (FiatShamir.squeezeMany Transcript.piCcsOracle.transcript
          ((ProductionKey.key relation ajtai).publicInputState running fresh)
          (FiatShamir.alphaLabels productionShape)).2 .gamma).2 =
      run Transcript.initialState (preRoundCalls running fresh) := by
  simp only [oracle_squeeze_eq, squeezeMany_state, publicInputState_eq_run, preRoundCalls,
    ← run_append, List.append_assoc]

/-- Seal for the SumCheck point: round `round` reads the state after every
earlier round, its own message, and its label. -/
theorem round_seal
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof (ProductionKey.degreeBound relation))
    (round : Fin productionShape.cubeVariables) :
    ((ProductionKey.key relation ajtai).piCcsExecution running fresh
        proof).coins.roundPoint.coordinates.getD round.val K.zero =
      readK (run Transcript.initialState (roundPrefixCalls running fresh proof round ++
        absorbCalls (Transcript.labelWord (.sumcheck round)))) := by
  rw [Nifs.PaperNonInteractive.Key.piCcsExecution_coins_eq_derive]
  change (FiatShamir.deriveRoundsFrom Transcript.piCcsOracle.transcript (messages proof)
      (Transcript.piCcsOracle.transcript.squeeze
        (FiatShamir.squeezeMany Transcript.piCcsOracle.transcript
          ((ProductionKey.key relation ajtai).publicInputState running fresh)
          (FiatShamir.alphaLabels productionShape)).2 .gamma).2
      (canonicalFinIndices productionShape.cubeVariables)).1.getD round.val K.zero = _
  rw [deriveRoundsFrom_getD _ _ _ _ (by simp [canonicalFinIndices]),
    preRoundState_eq_run, ← run_append, roundPrefixCalls, canonicalFinIndices_getElem]
  simp only [Fin.eta, List.append_assoc]

/-- The state after the last SumCheck round, before `y′`. -/
private theorem roundsState_eq_run
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof (ProductionKey.degreeBound relation)) :
    ((ProductionKey.key relation ajtai).piCcsExecution running fresh
        proof).coins.finalState =
      run Transcript.initialState (preRoundCalls running fresh ++
        (canonicalFinIndices productionShape.cubeVariables).flatMap
          (roundCalls (messages proof))) := by
  unfold messages
  rewrite [Nifs.PaperNonInteractive.Key.piCcsExecution_coins_eq_derive, derive_finalState,
    derivePreSumcheck_state, ProductionKey.key_oracle_eq,
    Transcript.piCcsOracle.initialState_is_prior]
  dsimp only
  rw [deriveRoundsFrom_state, preRoundState_eq_run, ← run_append]

/-- The `Π_CCS` outgoing state is the run of `outputCalls`. -/
private theorem outgoingState_eq_run
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof (ProductionKey.degreeBound relation)) :
    ((ProductionKey.key relation ajtai).piCcsExecution running fresh proof).outgoingState =
      run Transcript.initialState (outputCalls running fresh proof) := by
  rw [Nifs.PaperNonInteractive.Key.piCcsExecution_outgoingState_eq_absorbPiCcsOutput,
    ProductionKey.key_absorbPiCcsOutput, absorbFullOutput_eq_run, roundsState_eq_run,
    ← run_append, outputCalls]

/-- Seal for `Π_RLC`: scalar `index` reads the state after the complete `y′`,
the earlier scalar domains, and its own domain chunk. -/
theorem rho_seal
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof (ProductionKey.degreeBound relation)) :
    (ProductionKey.key relation ajtai).piRlcChallenges running fresh proof =
      some fun index => readRho (run Transcript.initialState
        (outputCalls running fresh proof ++ rhoReadCalls index.val)) := by
  unfold Nifs.PaperNonInteractive.Key.piRlcChallenges
  show ProductionKey.piRlcResponse
      ((ProductionKey.key relation ajtai).piCcsExecution running fresh proof).outgoingState = _
  rw [ProductionKey.piRlcResponse, Transcript.PiRlcSampler.piRlcChallenges,
    Transcript.PiRlcSampler.piRlcChallengesWithState_challenges, outgoingState_eq_run]
  refine congrArg some (funext fun index => ?_)
  simp only [Transcript.PiRlcSampler.sampleRingChallenge,
    Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.challengeAt,
    Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.scalarAt,
    Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.drawAt, readRho, rhoReadCalls,
    run_append, rhoStateAt_eq_run, rhoEnter_eq_step, run_cons, run_nil]

/-! ## Coverage -/

/-- Equal flat maps with equal piece lengths agree piece by piece. -/
private theorem flatMap_eq_on {α β : Type*} (indices : List α) (left right : α → List β)
    (lengths : ∀ index ∈ indices, (left index).length = (right index).length)
    (same : indices.flatMap left = indices.flatMap right) :
    ∀ index ∈ indices, left index = right index := by
  induction indices with
  | nil => simp
  | cons head tail inductionHypothesis =>
      simp only [List.flatMap_cons] at same
      obtain ⟨headEqual, tailEqual⟩ := List.append_inj same (lengths head (by simp))
      intro index member
      rcases List.mem_cons.mp member with rfl | member
      · exact headEqual
      · exact inductionHypothesis
          (fun index member => lengths index (List.mem_cons_of_mem _ member))
          tailEqual index member

private theorem block_injective : Function.Injective block :=
  fun _ _ same => (List.cons.inj same).2

private theorem serializeK_injective : Function.Injective serializeK := by
  intro left right same
  cases left
  cases right
  simp only [serializeK, List.cons.injEq, and_true] at same
  obtain ⟨rfl, rfl⟩ := same
  rfl

private theorem serializeKs_length (values : List K) :
    (values.flatMap serializeK).length = 2 * values.length := by
  induction values with
  | nil => rfl
  | cons value values inductionHypothesis =>
      simp only [List.flatMap_cons, List.length_append, inductionHypothesis]
      simp [serializeK]
      omega

private theorem serializeKs_injective {left right : List K}
    (same : left.flatMap serializeK = right.flatMap serializeK) : left = right := by
  induction left generalizing right with
  | nil =>
      cases right with
      | nil => rfl
      | cons _ _ => simp [serializeK] at same
  | cons value values inductionHypothesis =>
      cases right with
      | nil => simp [serializeK] at same
      | cons other others =>
          simp only [List.flatMap_cons] at same
          obtain ⟨headEqual, tailEqual⟩ := List.append_inj same rfl
          rw [serializeK_injective headEqual, inductionHypothesis tailEqual]

private theorem fixedPolynomial_eq {degree : Nat}
    {left right : SumCheck.Finite.FixedPolynomial K degree}
    (same : left.coefficients = right.coefficients) : left = right := by
  cases left
  cases right
  cases same
  rfl

private theorem messageCalls_length {degree : Nat} (left right : Proof degree)
    (round : Fin productionShape.cubeVariables) :
    (messageCalls (messages left) round).length =
      (messageCalls (messages right) round).length := by
  simp [messageCalls, messages, block, Transcript.serializeMessage,
    SumCheck.Finite.FixedPolynomial.toMessage, (left.piCcsRounds round).coefficients_length,
    (right.piCcsRounds round).coefficients_length]

private theorem messageCalls_injective {degree : Nat} {left right : Proof degree}
    {round : Fin productionShape.cubeVariables}
    (same : messageCalls (messages left) round = messageCalls (messages right) round) :
    left.piCcsRounds round = right.piCcsRounds round := by
  have words := block_injective (absorbCalls_injective (by
    simp [block, Transcript.serializeMessage, messages,
      SumCheck.Finite.FixedPolynomial.toMessage, (left.piCcsRounds round).coefficients_length,
      (right.piCcsRounds round).coefficients_length]) same)
  exact fixedPolynomial_eq (serializeKs_injective (List.cons.inj words).2)

private theorem roundCalls_length {degree : Nat} (left right : Proof degree)
    (round : Fin productionShape.cubeVariables) :
    (roundCalls (messages left) round).length =
      (roundCalls (messages right) round).length := by
  simp only [roundCalls, List.length_append, messageCalls_length left right round]

private theorem roundCalls_injective {degree : Nat} {left right : Proof degree}
    {round : Fin productionShape.cubeVariables}
    (same : roundCalls (messages left) round = roundCalls (messages right) round) :
    left.piCcsRounds round = right.piCcsRounds round :=
  messageCalls_injective (List.append_inj same (messageCalls_length left right round)).1

/-- Flat maps with equal piece lengths have equal lengths. -/
private theorem flatMap_length_eq {α β : Type*} (indices : List α) (left right : α → List β)
    (lengths : ∀ index ∈ indices, (left index).length = (right index).length) :
    (indices.flatMap left).length = (indices.flatMap right).length := by
  rw [List.length_flatMap, List.length_flatMap]
  congr 1
  exact List.map_congr_left lengths

private theorem roundsCalls_length {degree : Nat} (left right : Proof degree)
    (indices : List (Fin productionShape.cubeVariables)) :
    (indices.flatMap (roundCalls (messages left))).length =
      (indices.flatMap (roundCalls (messages right))).length := by
  exact flatMap_length_eq _ _ _ fun round _ => roundCalls_length left right round

private theorem serializeCommitment_injective : Function.Injective serializeCommitment := by
  intro left right same
  funext row coefficient
  have entry := congrArg
    (fun words => words.getD (row.val * ringDegree + coefficient.val) 0) same
  simpa only [PiCCSRepresentation.serializeCommitment_getD] using entry

private theorem outputWords_injective
    {left right : FullOutputCoordinates.FullOutput K productionShape}
    (same : outputWords left = outputWords right) : left = right := by
  have sources := flatMap_eq_on _ _ _ (fun _ _ => by simp [serializeK]) same
  have parts := fun source => List.append_inj
    (sources source (List.mem_finRange source)) (by simp [serializeK])
  have pad : left.padCoordinate = right.padCoordinate := by
    funext source coefficient
    exact serializeK_injective (flatMap_eq_on _
      (fun coefficient => serializeK (left.padCoordinate source coefficient))
      (fun coefficient => serializeK (right.padCoordinate source coefficient))
      (fun _ _ => rfl) (parts source).1 coefficient (List.mem_finRange _))
  have matrix : left.matrixCoordinate = right.matrixCoordinate := by
    funext source matrix coefficient
    have matrices := flatMap_eq_on _ _ _ (fun _ _ => by simp [serializeK]) (parts source).2
      matrix (List.mem_finRange _)
    exact serializeK_injective (flatMap_eq_on _
      (fun coefficient => serializeK (left.matrixCoordinate source matrix coefficient))
      (fun coefficient => serializeK (right.matrixCoordinate source matrix coefficient))
      (fun _ _ => rfl) matrices coefficient (List.mem_finRange _))
  cases left
  cases right
  simp only at pad matrix
  rw [pad, matrix]

private theorem mem_take_canonicalFinIndices {count : Nat} (index : Fin count) {bound : Nat}
    (below : index.val < bound) : index ∈ (canonicalFinIndices count).take bound := by
  rw [List.mem_iff_getElem]
  exact ⟨index.val, by simp [canonicalFinIndices]; omega, by simp [canonicalFinIndices]⟩

private theorem mem_canonicalFinIndices {count : Nat} (index : Fin count) :
    index ∈ canonicalFinIndices count := by
  simp [canonicalFinIndices]

private theorem serializePublicInput_injective :
    Function.Injective
      (serializePublicInput (logicalWidth := logicalWidth) (publicFits := publicFits)) := by
  intro left right same
  funext column
  exact List.map_inj_left.mp same column (List.mem_finRange column)

private theorem statementCalls_length
    (running running' : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh fresh' : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (statementCalls running fresh).length = (statementCalls running' fresh').length := by
  simp only [statementCalls, ProductionKey.publicInputBlocks, List.flatMap_append,
    List.flatMap_cons, List.flatMap_nil, List.append_nil, List.flatMap_assoc, List.length_append]
  have digestLength : (absorbCalls (block (ProductionKey.priorDigest fresh))).length =
      (absorbCalls (block (ProductionKey.priorDigest fresh'))).length := by
    simp [ProductionKey.priorDigest, decodeHash]
  rw [digestLength, flatMap_length_eq _ _
    (fun index => absorbCalls (block (serializeCommitment (fresh'.commitments index))) ++
      absorbCalls (block (serializePublicInput (fresh'.publicInputs index))))
    (fun _ _ => by simp)]

/-- Coverage before `α` and `γ`: equal statement calls give equal fresh
statements. -/
theorem statementCalls_identify_fresh
    {running running' : Running (logicalWidth := logicalWidth) (publicFits := publicFits)}
    {fresh fresh' : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)}
    (same : statementCalls running fresh = statementCalls running' fresh') :
    fresh = fresh' := by
  have blocks := List.append_cancel_left same
  simp only [ProductionKey.publicInputBlocks, List.flatMap_append, List.flatMap_cons,
    List.flatMap_nil, List.append_nil, List.flatMap_assoc] at blocks
  obtain ⟨_, perFresh⟩ := List.append_inj blocks
    (by simp [ProductionKey.priorDigest, decodeHash])
  have each := flatMap_eq_on _ _ _ (fun _ _ => by simp) perFresh
  have parts := fun index => List.append_inj (each index (List.mem_finRange index))
    (by simp)
  have commitments : fresh.commitments = fresh'.commitments := by
    funext index
    exact serializeCommitment_injective
      (block_injective (absorbCalls_injective (by simp) (parts index).1))
  have publicInputs : fresh.publicInputs = fresh'.publicInputs := by
    funext index
    exact serializePublicInput_injective
      (block_injective (absorbCalls_injective (by simp) (parts index).2))
  cases fresh
  cases fresh'
  simp only at commitments publicInputs
  rw [commitments, publicInputs]

/-- Coverage before a SumCheck challenge: equal calls give equal fresh
statements and equal messages for this and every earlier round. -/
theorem roundPrefixCalls_identify {degree : Nat}
    {running running' : Running (logicalWidth := logicalWidth) (publicFits := publicFits)}
    {fresh fresh' : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)}
    {proof proof' : Proof degree} {round : Fin productionShape.cubeVariables}
    (same : roundPrefixCalls running fresh proof round =
      roundPrefixCalls running' fresh' proof' round) :
    fresh = fresh' ∧ ∀ earlier : Fin productionShape.cubeVariables,
      earlier.val ≤ round.val → proof.piCcsRounds earlier = proof'.piCcsRounds earlier := by
  simp only [roundPrefixCalls, preRoundCalls, List.append_assoc] at same
  obtain ⟨statementEqual, afterStatement⟩ :=
    List.append_inj same (statementCalls_length running running' fresh fresh')
  have afterLabels := List.append_cancel_left (List.append_cancel_left afterStatement)
  obtain ⟨earlierEqual, ownEqual⟩ :=
    List.append_inj afterLabels (roundsCalls_length proof proof' _)
  refine ⟨statementCalls_identify_fresh statementEqual, fun earlier atMost => ?_⟩
  rcases Nat.lt_or_eq_of_le atMost with before | equal
  · exact roundCalls_injective (flatMap_eq_on _ _ _
      (fun round _ => roundCalls_length proof proof' round) earlierEqual earlier
      (mem_take_canonicalFinIndices earlier before))
  · rw [Fin.ext equal]
    exact messageCalls_injective ownEqual

/-- Coverage before `Π_RLC`: equal calls give equal fresh statements, every
SumCheck message, and the complete `y′`. -/
theorem outputCalls_identify {degree : Nat}
    {running running' : Running (logicalWidth := logicalWidth) (publicFits := publicFits)}
    {fresh fresh' : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)}
    {proof proof' : Proof degree}
    (same : outputCalls running fresh proof = outputCalls running' fresh' proof') :
    fresh = fresh' ∧ proof.piCcsRounds = proof'.piCcsRounds ∧
      proof.piCcsOutput = proof'.piCcsOutput := by
  simp only [outputCalls, preRoundCalls, List.append_assoc] at same
  obtain ⟨statementEqual, afterStatement⟩ :=
    List.append_inj same (statementCalls_length running running' fresh fresh')
  have afterLabels := List.append_cancel_left (List.append_cancel_left afterStatement)
  obtain ⟨roundsEqual, outputEqual⟩ :=
    List.append_inj afterLabels (roundsCalls_length proof proof' _)
  refine ⟨statementCalls_identify_fresh statementEqual, ?_, ?_⟩
  · funext round
    exact roundCalls_injective (flatMap_eq_on _ _ _
      (fun round _ => roundCalls_length proof proof' round) roundsEqual round
      (mem_canonicalFinIndices round))
  · exact outputWords_injective (block_injective (absorbCalls_injective
      (by simp [outputWords, serializeK, List.length_flatMap]) outputEqual))

end Schedule

/-! ## Contract -/

/-- Every verifier challenge of one production NIFS execution. -/
inductive Challenge where
  | alpha (coordinate : Fin productionShape.cubeVariables)
  | gamma
  | round (index : Fin productionShape.cubeVariables)
  | rho (index : Fin (Nifs.PaperProfile.arity).total)

section Contract

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- The prover-dependent calls before a challenge. The seal theorems show that
only fixed labels, squeezes, and `Π_RLC` domain chunks follow before the read. -/
def proverCalls {degree : Nat}
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof degree) : Challenge → List Call
  | .alpha _ => statementCalls running fresh
  | .gamma => statementCalls running fresh
  | .round index => roundPrefixCalls running fresh proof index
  | .rho _ => outputCalls running fresh proof

/-- Two executions agree on everything the verifier has received before a
challenge. -/
def Received {degree : Nat}
    (fresh fresh' : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof proof' : Proof degree) : Challenge → Prop
  | .alpha _ => fresh = fresh'
  | .gamma => fresh = fresh'
  | .round index => fresh = fresh' ∧ ∀ earlier : Fin productionShape.cubeVariables,
      earlier.val ≤ index.val → proof.piCcsRounds earlier = proof'.piCcsRounds earlier
  | .rho _ => fresh = fresh' ∧ proof.piCcsRounds = proof'.piCcsRounds ∧
      proof.piCcsOutput = proof'.piCcsOutput

theorem Received.fresh_eq {degree : Nat}
    {fresh fresh' : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)}
    {proof proof' : Proof degree} {challenge : Challenge}
    (received : Received fresh fresh' proof proof' challenge) : fresh = fresh' := by
  cases challenge with
  | alpha => exact received
  | gamma => exact received
  | round => exact received.1
  | rho => exact received.1

/-- Equal prover-dependent calls before a challenge give equal received data. -/
theorem proverCalls_identify {degree : Nat}
    {running running' : Running (logicalWidth := logicalWidth) (publicFits := publicFits)}
    {fresh fresh' : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)}
    {proof proof' : Proof degree} (challenge : Challenge)
    (same : proverCalls running fresh proof challenge =
      proverCalls running' fresh' proof' challenge) :
    Received fresh fresh' proof proof' challenge := by
  cases challenge with
  | alpha => exact statementCalls_identify_fresh same
  | gamma => exact statementCalls_identify_fresh same
  | round => exact roundPrefixCalls_identify same
  | rho => exact outputCalls_identify same

/-- The HyperNova prior-state link. The fresh public input carries the hash of
the prior preimage, and the NIFS running statement is the preimage's running
vector. `ActualTerminalSecurity.terminal_implies_nifsOrBaseOrCollision`
derives the digest equation from terminal acceptance. -/
structure PriorLink
    (prior : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)) : Prop where
  digest : ProductionKey.priorDigest fresh = stateHash (publicFits := publicFits) prior
  running_eq : prior.running functionIndex = running
  wellFormed : StateEncoding.WellFormed prior

/-- The coverage contract. If two executions present equal prover-dependent
duplex calls before a challenge, they agree on the prior preimage (verifier
key, iteration, application states, program counter), the running statement,
the fresh statement, and every earlier prover message, unless the state hash
collides. -/
theorem calls_identify_view_or_collision {degree : Nat} (challenge : Challenge)
    {prior prior' : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits)}
    {running running' : Running (logicalWidth := logicalWidth) (publicFits := publicFits)}
    {fresh fresh' : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)}
    {proof proof' : Proof degree}
    (link : PriorLink prior running fresh) (link' : PriorLink prior' running' fresh')
    (same : proverCalls running fresh proof challenge =
      proverCalls running' fresh' proof' challenge) :
    (prior = prior' ∧ running = running' ∧ Received fresh fresh' proof proof' challenge) ∨
      PiCCSSecurity.StateHashCollision prior prior' := by
  have received := proverCalls_identify challenge same
  have digestEqual : stateHash (publicFits := publicFits) prior =
      stateHash (publicFits := publicFits) prior' := by
    rw [← link.digest, ← link'.digest, received.fresh_eq]
  rcases PiCCSSecurity.stateHash_identifies_statement_or_collision prior prior'
      link.wellFormed link'.wellFormed digestEqual with priorEqual | collision
  · refine Or.inl ⟨priorEqual, ?_, received⟩
    rw [← link.running_eq, ← link'.running_eq, priorEqual]
  · exact Or.inr collision

end Contract

end NightstreamFPrime.Layout.Stage1.TranscriptCoverage

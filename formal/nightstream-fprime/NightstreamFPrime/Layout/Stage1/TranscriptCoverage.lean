import NightstreamFPrime.Layout.Stage1.PiCCSSecurity
import NightstreamFPrime.Lifecycle.ProductionKey
import NightstreamFPrime.Spec.Folding.PiCCS.Transcript
import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.TranscriptHistory

/-!
Owns the Fiat–Shamir coverage contract of the production NIFS transcript.

Every verifier challenge (`α`, `γ`, the SumCheck points, and the `Π_RLC`
scalars) is a read of the Poseidon2 state after an explicit list of
permutation inputs. One input is one zero-padded rate chunk, and a squeeze is
the zero chunk, so a list of inputs is exactly what the permutation sees.

Inputs: the production key, the fresh statement, the NIFS proof, and the prior
hash preimage.

Outputs:
- `challenge_seal`: each key challenge reads the state after
  `proverCalls c ++ fixedCalls c`, and `fixedCalls` takes no prover data;
- `coins_eq_reads`: the PiCCS coin record is exactly these reads, so a new
  coin field breaks the statement;
- `proverCalls_identify`: equal prover-dependent calls give equal fresh
  statements and equal earlier prover messages; before `Π_RLC` they give equal
  proofs up to the `Π_DEC` child messages, so a new proof field breaks it;
- `calls_identify_view_or_collision`: with the prior-state link, equal calls
  also identify the prior preimage (verifier-key digest, iteration,
  application states, program counter, running statement), unless the state
  hash collides. `ActualTerminalSecurity.terminal_calls_identify_view_or_collision`
  applies it to accepted terminals.

The `Π_RLC` reads use the query keys of the sampler's oracle model
(`ScheduleLaw.queryAt`).

Invariant: removing an absorption of verifier-relevant data, moving it after
a challenge that depends on it, or making its encoding ambiguous breaks
`challenge_seal` or a coverage proof.

Does not own: Poseidon2 security, the Fiat–Shamir transfer (an approved
external assumption), or the circuit refinement of this schedule. The running
statement enters only through the prior digest. The `Π_DEC` child messages
follow the last challenge; the next state hash binds them.
-/

namespace NightstreamFPrime.Layout.Stage1.TranscriptCoverage

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

/-! ## Permutation inputs -/

/-- One permutation input: a zero-padded rate chunk added to the state. -/
abbrev Call := Fin Poseidon2.rate → F

/-- A squeeze adds nothing before the permutation. -/
def squeeze : Call := fun _ => 0

/-- Add one chunk to the state, then permute. -/
def step (state : Transcript.State) (chunk : Call) : Transcript.State :=
  Poseidon2.absorbBlock state (List.ofFn chunk)

/-- Apply a list of permutation inputs in order. -/
def run (state : Transcript.State) (calls : List Call) : Transcript.State :=
  calls.foldl step state

@[simp] private theorem run_nil (state : Transcript.State) : run state [] = state := rfl

@[simp] private theorem run_cons (state : Transcript.State) (call : Call) (calls : List Call) :
    run state (call :: calls) = run (step state call) calls := rfl

private theorem run_append (state : Transcript.State) (left right : List Call) :
    run state (left ++ right) = run (run state left) right := by
  simp [run, List.foldl_append]

private theorem run_length (state : Transcript.State) {calls : List Call}
    (nonempty : calls ≠ []) : (run state calls).length = Poseidon2.width := by
  induction calls generalizing state with
  | nil => exact absurd rfl nonempty
  | cons call calls inductionHypothesis =>
      rw [run_cons]
      by_cases empty : calls = []
      · subst empty
        rw [run_nil]
        exact Poseidon2.absorbBlock_length _ _
      · exact inductionHypothesis _ empty

/-- On a full-width state, a squeeze is the plain permutation. -/
private theorem step_squeeze (state : Transcript.State)
    (fixed : state.length = Poseidon2.width) :
    step state squeeze = Poseidon2.permute state :=
  Poseidon2.absorbBlock_zero state fixed fun index => by
    simp only [squeeze, List.getD_eq_getElem?_getD, List.getElem?_ofFn]
    split <;> rfl

/-- The rate chunk that the sponge adds for a short word list. -/
def padChunk (words : List F) : Call :=
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

/-- Equal-length word lists with equal padded chunks are equal: zero padding
creates no ambiguity when the length is fixed. -/
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
    Nat.mod_lt _ (show 0 < Poseidon2.rate by decide), if_true,
    List.getElem?_drop, split] at lane
  simpa [List.getElem?_eq_getElem leftBound, List.getElem?_eq_getElem rightBound]
    using lane

/-! ## Labelled squeezes and SumCheck rounds -/

/-- The extension challenge that `Transcript.squeezeK` reads before its two
squeezes. -/
def readK (state : Transcript.State) : K :=
  ⟨state.getD 0 0, (Poseidon2.permute state).getD 0 0⟩

private theorem squeezeK_eq (state : Transcript.State)
    (fixed : state.length = Poseidon2.width) :
    Transcript.squeezeK state = (readK state, run state [squeeze, squeeze]) := by
  simp only [Transcript.squeezeK, Transcript.squeezeF, readK, run_cons, run_nil]
  rw [step_squeeze state fixed, step_squeeze _ (Poseidon2.permute_length state)]

/-- One labelled PiCCS squeeze: absorb the label words, then squeeze twice. -/
def labelCalls (label : FiatShamir.ChallengeLabel productionShape) : List Call :=
  absorbCalls (Transcript.labelWord label) ++ [squeeze, squeeze]

private theorem labelWord_ne_nil (label : FiatShamir.ChallengeLabel productionShape) :
    Transcript.labelWord label ≠ [] := by
  cases label <;> simp [Transcript.labelWord]

private theorem oracle_squeeze_eq (state : Transcript.State)
    (label : FiatShamir.ChallengeLabel productionShape) :
    Transcript.piCcsOracle.transcript.squeeze state label =
      (readK (run state (absorbCalls (Transcript.labelWord label))),
        run state (labelCalls label)) := by
  change Transcript.squeezeK (Transcript.absorb state (Transcript.labelWord label)) = _
  rw [absorb_eq_run, squeezeK_eq _ (run_length _ (absorbCalls_ne_nil (labelWord_ne_nil label))),
    labelCalls, run_append]

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

/-! ## Production schedule -/

section Schedule

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- Calls before the first `α` label: the digest-only domain tag and the key's
public-input blocks. The running statement is not an argument. -/
def statementCalls
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)) : List Call :=
  absorbCalls Transcript.piCcsDigestDomainTag ++
    (ProductionKey.publicInputBlocks fresh).flatMap fun words => absorbCalls (block words)

/-- Calls before the first SumCheck message: the statement, then every
labelled `α` squeeze and the `γ` squeeze. -/
def preRoundCalls
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)) : List Call :=
  statementCalls fresh ++
    (FiatShamir.alphaLabels productionShape).flatMap labelCalls ++ labelCalls .gamma

/-- Prover-dependent calls before the round `round` challenge: every earlier
round and the round's own message. -/
def roundPrefixCalls {degree : Nat}
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof degree) (round : Fin productionShape.cubeVariables) : List Call :=
  preRoundCalls fresh ++
    ((canonicalFinIndices productionShape.cubeVariables).take round.val).flatMap
      (roundCalls (messages proof)) ++
    messageCalls (messages proof) round

/-- Calls before the first `Π_RLC` scalar: every round and the complete `y′`. -/
def outputCalls {degree : Nat}
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof degree) : List Call :=
  preRoundCalls fresh ++
    (canonicalFinIndices productionShape.cubeVariables).flatMap (roundCalls (messages proof)) ++
    absorbCalls (block (ProductionKey.fullOutputWords proof.piCcsOutput))

end Schedule

/-- A four-lane oracle block of the `Π_RLC` sampler as a permutation input. -/
def widen (draw : Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Draw) : Call := fun lane =>
  if inside : lane.val < Spec.Folding.Nifs.NonInteractive.PiRlcSampler.drawWidth then
    draw ⟨lane.val, inside⟩
  else 0

/-- The sampler's scalar index of one `Π_RLC` challenge. -/
def rhoIndex (index : Fin (Nifs.PaperProfile.arity).total) : Fin 17 :=
  ⟨index.val, index.isLt.trans_eq (by decide)⟩

/-- The `Π_RLC` challenge that the sampler reads from four state lanes. -/
def readRho (state : Transcript.State) : RingF :=
  Phi81StrongSet.embedScalar
    (Spec.Folding.Nifs.NonInteractive.PiRlcSampler.sample
      (Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.block state))

private theorem step_widen (state : Transcript.State)
    (draw : Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Draw) :
    step state (widen draw) = Poseidon2.absorbBlock state (List.ofFn draw) := by
  unfold step Poseidon2.absorbBlock
  -- The lanes past `drawWidth` are zero in both chunks; `congr` closes this by evaluation.
  congr 1

/-- The oracle model's replay is the same run of permutation inputs. -/
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

/-! ## Seals -/

/-- Every verifier challenge of one production NIFS execution. -/
inductive Challenge where
  | alpha (coordinate : Fin productionShape.cubeVariables)
  | gamma
  | round (index : Fin productionShape.cubeVariables)
  | rho (index : Fin (Nifs.PaperProfile.arity).total)

/-- The value type of a challenge. -/
abbrev Challenge.Value : Challenge → Type
  | .alpha _ => K
  | .gamma => K
  | .round _ => K
  | .rho _ => RingF

/-- How a challenge reads the sponge state. -/
def Challenge.read : (challenge : Challenge) → Transcript.State → challenge.Value
  | .alpha _ => readK
  | .gamma => readK
  | .round _ => readK
  | .rho _ => readRho

/-- The calls between the prover-dependent calls and the read: labels,
squeezes, and `Π_RLC` domain chunks. The signature admits no prover data. -/
def fixedCalls : Challenge → List Call
  | .alpha coordinate =>
      ((FiatShamir.alphaLabels productionShape).take coordinate.val).flatMap labelCalls ++
        absorbCalls (Transcript.labelWord (.alpha coordinate))
  | .gamma =>
      (FiatShamir.alphaLabels productionShape).flatMap labelCalls ++
        absorbCalls (Transcript.labelWord .gamma)
  | .round index => absorbCalls (Transcript.labelWord (.sumcheck index))
  | .rho index =>
      (Spec.Folding.Nifs.NonInteractive.PiRlcSampler.ScheduleLaw.queryAt [] (rhoIndex index)).val.map
        widen

section Contract

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- The prover-dependent calls before a challenge. -/
def proverCalls {degree : Nat}
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof degree) : Challenge → List Call
  | .alpha _ => statementCalls fresh
  | .gamma => statementCalls fresh
  | .round index => roundPrefixCalls fresh proof index
  | .rho _ => outputCalls fresh proof

section Seals

variable (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
  (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
  (proof : Proof (ProductionKey.degreeBound relation))

/-- The production key's value for one challenge. -/
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

private theorem publicInputState_eq_run :
    (ProductionKey.key relation ajtai).publicInputState running fresh =
      run Transcript.initialState (statementCalls fresh) := by
  rw [ProductionKey.key_publicInputState_eq, absorbBlocks_eq_run, absorb_eq_run,
    statementCalls, run_append]

private theorem alpha_seal
    (coordinate : Fin productionShape.cubeVariables) :
    ((ProductionKey.key relation ajtai).piCcsExecution running fresh
        proof).coins.alpha.coordinates.getD coordinate.val K.zero =
      readK (run Transcript.initialState
        (statementCalls fresh ++ fixedCalls (.alpha coordinate))) := by
  rw [Nifs.PaperNonInteractive.Key.piCcsExecution_coins_eq_derive]
  change (FiatShamir.squeezeMany Transcript.piCcsOracle.transcript
      ((ProductionKey.key relation ajtai).publicInputState running fresh)
      (FiatShamir.alphaLabels productionShape)).1.getD coordinate.val K.zero = _
  rw [squeezeMany_getD _ _ _ (by simp [FiatShamir.alphaLabels_length]),
    publicInputState_eq_run, ← run_append]
  simp [fixedCalls, FiatShamir.alphaLabels, canonicalFinIndices]

private theorem gamma_seal :
    ((ProductionKey.key relation ajtai).piCcsExecution running fresh proof).coins.gamma =
      readK (run Transcript.initialState (statementCalls fresh ++ fixedCalls .gamma)) := by
  rw [Nifs.PaperNonInteractive.Key.piCcsExecution_coins_eq_derive]
  change (Transcript.piCcsOracle.transcript.squeeze
      (FiatShamir.squeezeMany Transcript.piCcsOracle.transcript
        ((ProductionKey.key relation ajtai).publicInputState running fresh)
        (FiatShamir.alphaLabels productionShape)).2 .gamma).1 = _
  simp only [oracle_squeeze_eq, squeezeMany_state, publicInputState_eq_run, fixedCalls,
    ← run_append, List.append_assoc]

private theorem preRoundState_eq_run :
    (Transcript.piCcsOracle.transcript.squeeze
        (FiatShamir.squeezeMany Transcript.piCcsOracle.transcript
          ((ProductionKey.key relation ajtai).publicInputState running fresh)
          (FiatShamir.alphaLabels productionShape)).2 .gamma).2 =
      run Transcript.initialState (preRoundCalls fresh) := by
  simp only [oracle_squeeze_eq, squeezeMany_state, publicInputState_eq_run, preRoundCalls,
    ← run_append, List.append_assoc]

private theorem round_seal
    (round : Fin productionShape.cubeVariables) :
    ((ProductionKey.key relation ajtai).piCcsExecution running fresh
        proof).coins.roundPoint.coordinates.getD round.val K.zero =
      readK (run Transcript.initialState
        (roundPrefixCalls fresh proof round ++ fixedCalls (.round round))) := by
  rw [Nifs.PaperNonInteractive.Key.piCcsExecution_coins_eq_derive]
  change (FiatShamir.deriveRoundsFrom Transcript.piCcsOracle.transcript (messages proof)
      (Transcript.piCcsOracle.transcript.squeeze
        (FiatShamir.squeezeMany Transcript.piCcsOracle.transcript
          ((ProductionKey.key relation ajtai).publicInputState running fresh)
          (FiatShamir.alphaLabels productionShape)).2 .gamma).2
      (canonicalFinIndices productionShape.cubeVariables)).1.getD round.val K.zero = _
  rw [deriveRoundsFrom_getD _ _ _ _ (by simp [canonicalFinIndices]),
    preRoundState_eq_run, ← run_append, roundPrefixCalls, canonicalFinIndices_getElem]
  simp only [fixedCalls, Fin.eta, List.append_assoc]

private theorem roundsState_eq_run :
    ((ProductionKey.key relation ajtai).piCcsExecution running fresh
        proof).coins.finalState =
      run Transcript.initialState (preRoundCalls fresh ++
        (canonicalFinIndices productionShape.cubeVariables).flatMap
          (roundCalls (messages proof))) := by
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
    ProductionKey.key_absorbPiCcsOutput, ProductionKey.absorbFullOutput, absorbBlock_eq_run,
    roundsState_eq_run, ← run_append, outputCalls]

private theorem rho_seal :
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
  -- `rfl` would compare `Fin 17` with `Fin arity.total` and evaluate the arity.
  simp only [Spec.Folding.Nifs.NonInteractive.PiRlcSampler.TranscriptHistory.replay,
    List.foldl_nil, rhoIndex]

/-- Seal: every key challenge reads the state after its prover-dependent
calls followed by fixed calls. A schedule change that moves prover data after
a challenge cannot satisfy this statement, because `fixedCalls` admits no
prover data. -/
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

/-- Guard: the PiCCS coin record is exactly the reads of `challenge_seal`. A new
coin field breaks this statement until it is covered. -/
theorem coins_eq_reads :
    ((ProductionKey.key relation ajtai).piCcsExecution running fresh proof).coins =
      { alpha := ⟨List.ofFn fun coordinate => readK (run Transcript.initialState
            (proverCalls fresh proof (.alpha coordinate) ++ fixedCalls (.alpha coordinate))),
          by simp⟩
        gamma := readK (run Transcript.initialState
          (proverCalls fresh proof .gamma ++ fixedCalls .gamma))
        roundPoint := ⟨List.ofFn fun index => readK (run Transcript.initialState
            (proverCalls fresh proof (.round index) ++ fixedCalls (.round index))),
          by simp⟩
        finalState := run Transcript.initialState (preRoundCalls fresh ++
          (canonicalFinIndices productionShape.cubeVariables).flatMap
            (roundCalls (messages proof))) } := by
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
  have alphaEqual : alpha = List.ofFn fun coordinate => readK (run Transcript.initialState
      (proverCalls fresh proof (.alpha coordinate) ++ fixedCalls (.alpha coordinate))) :=
    List.ext_getElem (by simp [alphaLength]) fun index inside _ =>
      (List.getD_eq_getElem _ K.zero inside).symm.trans
        ((alphaSeal ⟨index, alphaLength ▸ inside⟩).trans (by simp [proverCalls]))
  have pointEqual : point = List.ofFn fun index => readK (run Transcript.initialState
      (proverCalls fresh proof (.round index) ++ fixedCalls (.round index))) :=
    List.ext_getElem (by simp [pointLength]) fun index inside _ =>
      (List.getD_eq_getElem _ K.zero inside).symm.trans
        ((pointSeal ⟨index, pointLength ▸ inside⟩).trans (by simp [proverCalls]))
  subst alphaEqual pointEqual gammaSeal finalSeal
  rfl

end Seals

/-! ## Coverage -/

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
  exact SumCheck.Finite.FixedPolynomial.eq_of_coefficients
    (serializeKs_injective (List.cons.inj words).2)

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

private theorem roundsCalls_length {degree : Nat} (left right : Proof degree)
    (indices : List (Fin productionShape.cubeVariables)) :
    (indices.flatMap (roundCalls (messages left))).length =
      (indices.flatMap (roundCalls (messages right))).length :=
  flatMap_length_eq _ _ _ fun round _ => roundCalls_length left right round

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

private theorem statementCalls_length
    (fresh fresh' : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (statementCalls fresh).length = (statementCalls fresh').length := by
  simp only [statementCalls, ProductionKey.publicInputBlocks, List.flatMap_append,
    List.flatMap_cons, List.flatMap_nil, List.append_nil, List.flatMap_assoc, List.length_append]
  have digestLength : (absorbCalls (block (ProductionKey.priorDigest fresh))).length =
      (absorbCalls (block (ProductionKey.priorDigest fresh'))).length := by
    simp [ProductionKey.priorDigest, decodeHash]
  rw [digestLength, flatMap_length_eq _ _
    (fun index => absorbCalls (block (serializeCommitment (fresh'.commitments index))) ++
      absorbCalls (block (serializePublicInput (fresh'.publicInputs index))))
    (fun _ _ => by simp)]

private theorem statementCalls_identify_fresh
    {fresh fresh' : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)}
    (same : statementCalls fresh = statementCalls fresh') : fresh = fresh' := by
  have blocks := List.append_cancel_left same
  simp only [ProductionKey.publicInputBlocks, List.flatMap_append, List.flatMap_cons,
    List.flatMap_nil, List.append_nil, List.flatMap_assoc] at blocks
  obtain ⟨_, perFresh⟩ := List.append_inj blocks
    (by simp [ProductionKey.priorDigest, decodeHash])
  have each := flatMap_eq_of_lengths _ _ _ (fun _ _ => by simp) perFresh
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

private theorem roundPrefixCalls_identify {degree : Nat}
    {fresh fresh' : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)}
    {proof proof' : Proof degree} {round : Fin productionShape.cubeVariables}
    (same : roundPrefixCalls fresh proof round = roundPrefixCalls fresh' proof' round) :
    fresh = fresh' ∧ ∀ earlier : Fin productionShape.cubeVariables,
      earlier.val ≤ round.val → proof.piCcsRounds earlier = proof'.piCcsRounds earlier := by
  simp only [roundPrefixCalls, preRoundCalls, List.append_assoc] at same
  obtain ⟨statementEqual, afterStatement⟩ :=
    List.append_inj same (statementCalls_length fresh fresh')
  have afterLabels := List.append_cancel_left (List.append_cancel_left afterStatement)
  obtain ⟨earlierEqual, ownEqual⟩ :=
    List.append_inj afterLabels (roundsCalls_length proof proof' _)
  refine ⟨statementCalls_identify_fresh statementEqual, fun earlier atMost => ?_⟩
  rcases Nat.lt_or_eq_of_le atMost with before | equal
  · exact roundCalls_injective (flatMap_eq_of_lengths _ _ _
      (fun round _ => roundCalls_length proof proof' round) earlierEqual earlier
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
  simp only [outputCalls, preRoundCalls, List.append_assoc] at same
  obtain ⟨statementEqual, afterStatement⟩ :=
    List.append_inj same (statementCalls_length fresh fresh')
  have afterLabels := List.append_cancel_left (List.append_cancel_left afterStatement)
  obtain ⟨roundsEqual, outputEqual⟩ :=
    List.append_inj afterLabels (roundsCalls_length proof proof' _)
  have rounds : proof.piCcsRounds = proof'.piCcsRounds := funext fun round =>
    roundCalls_injective (flatMap_eq_of_lengths _ _ _
      (fun round _ => roundCalls_length proof proof' round) roundsEqual round
      (mem_canonicalFinIndices round))
  have output : proof.piCcsOutput = proof'.piCcsOutput :=
    fullOutputWords_injective (block_injective (absorbCalls_injective
      (by simp [ProductionKey.fullOutputWords, List.length_flatMap]) outputEqual))
  refine ⟨statementCalls_identify_fresh statementEqual, ?_⟩
  cases proof
  cases proof'
  simp only at rounds output
  subst rounds output
  rfl

/-! ## Contract -/

/-- Two executions agree on everything the transcript absorbs directly before
a challenge: the fresh statement and the earlier prover messages. Before
`Π_RLC` the proofs are equal except for the `Π_DEC` child messages, which follow
the last challenge. The running statement enters only through the prior
digest; see `PriorLink`. -/
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

private theorem AgreeOnAbsorbed.fresh_eq {degree : Nat}
    {fresh fresh' : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)}
    {proof proof' : Proof degree} {challenge : Challenge}
    (agree : AgreeOnAbsorbed fresh fresh' proof proof' challenge) : fresh = fresh' := by
  cases challenge with
  | alpha => exact agree
  | gamma => exact agree
  | round => exact agree.1
  | rho => exact agree.1

/-- Equal prover-dependent calls before a challenge give agreement on
everything the transcript absorbed directly. -/
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

/-- The HyperNova prior-state link: the fresh public input carries the hash of
the well-formed prior preimage, whose running vector is the NIFS running
statement. `ActualTerminalSecurity.terminal_implies_nifsOrBaseOrCollision`
supplies it for every accepted recursive terminal. -/
structure PriorLink
    (prior : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)) : Prop where
  digest : ProductionKey.priorDigest fresh = stateHash (publicFits := publicFits) prior
  wellFormed : StateEncoding.WellFormed prior

/-- The coverage contract. If two executions present equal prover-dependent
calls before a challenge, they agree on the prior preimage (verifier-key
digest, iteration, application states, program counter, running statement),
the fresh statement, and every earlier prover message, unless the state hash
collides. -/
theorem calls_identify_view_or_collision {degree : Nat} (challenge : Challenge)
    {prior prior' : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits)}
    {fresh fresh' : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)}
    {proof proof' : Proof degree}
    (link : PriorLink prior fresh) (link' : PriorLink prior' fresh')
    (same : proverCalls fresh proof challenge = proverCalls fresh' proof' challenge) :
    (prior = prior' ∧ AgreeOnAbsorbed fresh fresh' proof proof' challenge) ∨
      PiCCSSecurity.StateHashCollision prior prior' := by
  have agree := proverCalls_identify challenge same
  have digestEqual : stateHash (publicFits := publicFits) prior =
      stateHash (publicFits := publicFits) prior' := by
    rw [← link.digest, ← link'.digest, agree.fresh_eq]
  rcases PiCCSSecurity.stateHash_identifies_statement_or_collision prior prior'
      link.wellFormed link'.wellFormed digestEqual with priorEqual | collision
  · exact Or.inl ⟨priorEqual, agree⟩
  · exact Or.inr collision

end Contract

end NightstreamFPrime.Layout.Stage1.TranscriptCoverage

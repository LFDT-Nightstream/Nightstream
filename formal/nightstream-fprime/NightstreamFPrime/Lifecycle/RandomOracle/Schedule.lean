import Mathlib.Logic.Function.CompTypeclasses
import NightstreamFPrime.Lifecycle.RandomOracle.Sponge
import NightstreamFPrime.Spec.Folding.PiCCS.TranscriptReplay

/-! The challenge schedule of one fold transcript in the random-oracle model. The transcript
has the order of SuperNeo v1.2 Sections 7.3 and 7.4: a domain chunk, the statement words, the
Π_CCS coins `α` and `γ`, one coin after each Π_CCS round, the Π_CCS output words, and
`rhoCount` draws of the production Π_RLC sampler. A `Schedule` holds only what differs between
two such transcripts, so one proof serves the light fold and the refresh fold.

For each challenge, `challengeCalls` is the list of sponge calls before the read and
`Challenge.slot` is the lane pair that the read uses. Distinct challenges of one transcript
have distinct addresses (`challengeAddress_injective`), the address length does not depend on
the prover's data (`challengeCalls_length_eq`), and equal calls before a challenge fix the
absorbed data that the challenge depends on (`proverCalls_identify`). Transcripts with distinct
domain chunks share no sponge call prefix (`challengeCalls_separated`). A transcript that uses
`spongeOracle` reads its Π_CCS coins (`derive_eq_reads`, `derive_publicCoins`) and its Π_RLC
challenges (`rho_read`) at these addresses. `TranscriptCoverage` is the same schedule for the
one-circuit fold. -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.RandomOracle

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.RandomOracle.Sponge

/-- What differs between two fold transcripts. The statement words and the output words have
one length each, so the address of a challenge does not depend on the prover's data. The
production sampler schedule has 17 coordinates (`ScheduleLaw.queryAt`). -/
structure Schedule where
  shape : Shape
  domainChunk : List F
  domainChunk_length : domainChunk.length = Poseidon2.rate
  Statement : Type
  statementWords : Statement → List F
  statementWords_length : ∀ left right,
    (statementWords left).length = (statementWords right).length
  statementWords_injective : ∀ {left right},
    statementWords left = statementWords right → left = right
  Output : Type
  outputWords : Output → List F
  outputWords_length : ∀ left right, (outputWords left).length = (outputWords right).length
  outputWords_injective : ∀ {left right}, outputWords left = outputWords right → left = right
  rhoCount : Nat
  rhoCount_le : rhoCount ≤ 17

namespace Schedule

/-- The prover messages that the transcript absorbs after the statement. -/
structure Messages (schedule : Schedule) (width : Nat) where
  rounds : Fin schedule.shape.cubeVariables → SumCheck.Finite.FixedPolynomial K width
  output : schedule.Output

def Messages.certificate {schedule : Schedule} {width : Nat} (messages : schedule.Messages width) :
    FiatShamir.Certificate K schedule.shape where
  rounds := fun round => (messages.rounds round).toMessage

inductive Challenge (schedule : Schedule) where
  | alpha (coordinate : Fin schedule.shape.cubeVariables)
  | gamma
  | round (index : Fin schedule.shape.cubeVariables)
  | rho (index : Fin schedule.rhoCount)
  deriving DecidableEq, Fintype

abbrev Challenge.Value {schedule : Schedule} : schedule.Challenge → Type
  | .alpha _ => K
  | .gamma => K
  | .round _ => K
  | .rho _ => RingF

def Challenge.slot {schedule : Schedule} : schedule.Challenge → Slot
  | .alpha coordinate => slotOf coordinate.val
  | .gamma => slotOf schedule.shape.cubeVariables
  | .round _ => slotOf 0
  | .rho _ => slotOf 0

def Challenge.read {schedule : Schedule} :
    (challenge : schedule.Challenge) → Transcript.State → challenge.Value
  | .alpha coordinate => readK (slotOf coordinate.val)
  | .gamma => readK (slotOf schedule.shape.cubeVariables)
  | .round _ => readK (slotOf 0)
  | .rho _ => readRho

/-- The sponge calls that absorb one round polynomial of degree `width`. -/
def roundCallCount (width : Nat) : Nat :=
  (2 * (width + 1) + Poseidon2.rate - 1) / Poseidon2.rate

theorem roundCallCount_pos (width : Nat) : 0 < roundCallCount width := by
  unfold roundCallCount Poseidon2.rate
  omega

section Calls

variable (schedule : Schedule) {width : Nat}

def rhoIndex (index : Fin schedule.rhoCount) : Fin 17 :=
  ⟨index.val, Nat.lt_of_lt_of_le index.isLt schedule.rhoCount_le⟩

def roundCalls (messages : schedule.Messages width) (round : Fin schedule.shape.cubeVariables) :
    List Call :=
  absorbCalls (Transcript.serializeMessage (messages.rounds round).toMessage)

def statementCalls (statement : schedule.Statement) : List Call :=
  absorbCalls schedule.domainChunk ++ absorbCalls (schedule.statementWords statement)

def preRoundCalls (statement : schedule.Statement) : List Call :=
  schedule.statementCalls statement ++
    List.replicate ((schedule.shape.cubeVariables + 1) / Transcript.coinsPerChunk) squeeze

def roundPrefixCalls (statement : schedule.Statement) (messages : schedule.Messages width)
    (round : Fin schedule.shape.cubeVariables) : List Call :=
  schedule.preRoundCalls statement ++
    ((canonicalFinIndices schedule.shape.cubeVariables).take round.val).flatMap
      (schedule.roundCalls messages) ++
    schedule.roundCalls messages round

def roundsCalls (statement : schedule.Statement) (messages : schedule.Messages width) :
    List Call :=
  schedule.preRoundCalls statement ++
    (canonicalFinIndices schedule.shape.cubeVariables).flatMap (schedule.roundCalls messages)

def outputCalls (statement : schedule.Statement) (messages : schedule.Messages width) :
    List Call :=
  schedule.roundsCalls statement messages ++ absorbCalls (schedule.outputWords messages.output)

def fixedCalls : schedule.Challenge → List Call
  | .alpha coordinate => List.replicate (coordinate.val / Transcript.coinsPerChunk) squeeze
  | .gamma => List.replicate (schedule.shape.cubeVariables / Transcript.coinsPerChunk) squeeze
  | .round _ => []
  | .rho index =>
      (Spec.Folding.Nifs.NonInteractive.PiRlcSampler.ScheduleLaw.queryAt []
        (schedule.rhoIndex index)).val.map widen

def proverCalls (statement : schedule.Statement) (messages : schedule.Messages width) :
    schedule.Challenge → List Call
  | .alpha _ => schedule.statementCalls statement
  | .gamma => schedule.statementCalls statement
  | .round index => schedule.roundPrefixCalls statement messages index
  | .rho _ => schedule.outputCalls statement messages

def challengeCalls (statement : schedule.Statement) (messages : schedule.Messages width)
    (challenge : schedule.Challenge) : List Call :=
  schedule.proverCalls statement messages challenge ++ schedule.fixedCalls challenge

def challengeAddress (statement : schedule.Statement) (messages : schedule.Messages width)
    (challenge : schedule.Challenge) : List Call × Slot :=
  (schedule.challengeCalls statement messages challenge, challenge.slot)

def coinsFrom (read : List Call × Slot → K) (statement : schedule.Statement)
    (messages : schedule.Messages width) : StrongReduction.PublicCoins K schedule.shape where
  alpha := ⟨List.ofFn fun coordinate =>
    read (schedule.challengeAddress statement messages (.alpha coordinate)), List.length_ofFn⟩
  gamma := read (schedule.challengeAddress statement messages .gamma)
  roundPoint := ⟨List.ofFn fun index =>
    read (schedule.challengeAddress statement messages (.round index)), List.length_ofFn⟩

/-- SuperNeo v1.2 Sections 7.3–7.4 order: the absorbed data that each challenge depends on. -/
def AgreeOnAbsorbed (statement statement' : schedule.Statement)
    (messages messages' : schedule.Messages width) : schedule.Challenge → Prop
  | .alpha _ => statement = statement'
  | .gamma => statement = statement'
  | .round index => statement = statement' ∧ ∀ earlier : Fin schedule.shape.cubeVariables,
      earlier.val ≤ index.val → messages.rounds earlier = messages'.rounds earlier
  | .rho _ => statement = statement' ∧ messages = messages'

end Calls

section Lengths

variable (schedule : Schedule) {width : Nat}

private theorem serializeKs_length (values : List K) :
    (values.flatMap serializeK).length = 2 * values.length := by
  induction values with
  | nil => rfl
  | cons value values inductionHypothesis =>
      rw [List.flatMap_cons, List.length_append, inductionHypothesis, List.length_cons]
      simp only [serializeK, List.length_cons, List.length_nil]
      omega

private theorem roundWords_length (polynomial : SumCheck.Finite.FixedPolynomial K width) :
    (Transcript.serializeMessage polynomial.toMessage).length = 2 * (width + 1) := by
  rw [Transcript.serializeMessage, serializeKs_length]
  exact congrArg (2 * ·) polynomial.coefficients_length

theorem roundCalls_length (messages : schedule.Messages width)
    (round : Fin schedule.shape.cubeVariables) :
    (schedule.roundCalls messages round).length = roundCallCount width := by
  rw [roundCalls, absorbCalls_length, roundWords_length, roundCallCount]

private theorem takeRounds_length (messages : schedule.Messages width)
    (round : Fin schedule.shape.cubeVariables) :
    (((canonicalFinIndices schedule.shape.cubeVariables).take round.val).flatMap
      (schedule.roundCalls messages)).length = round.val * roundCallCount width := by
  rw [flatMap_length_constant _ _ _ (schedule.roundCalls_length messages), List.length_take,
    canonicalFinIndices_length, Nat.min_eq_left (Nat.le_of_lt round.isLt)]

private theorem allRounds_length (messages : schedule.Messages width) :
    ((canonicalFinIndices schedule.shape.cubeVariables).flatMap
      (schedule.roundCalls messages)).length =
      schedule.shape.cubeVariables * roundCallCount width := by
  rw [flatMap_length_constant _ _ _ (schedule.roundCalls_length messages),
    canonicalFinIndices_length]

theorem statementCalls_length (statement statement' : schedule.Statement) :
    (schedule.statementCalls statement).length = (schedule.statementCalls statement').length := by
  simp only [statementCalls, List.length_append, absorbCalls_length,
    schedule.statementWords_length statement statement']

/-- The length of a challenge's calls past the statement: `count` calls per round message and
`output` calls for the output words. -/
private def offset (count output : Nat) : schedule.Challenge → Nat
  | .alpha coordinate => coordinate.val / Transcript.coinsPerChunk
  | .gamma => schedule.shape.cubeVariables / Transcript.coinsPerChunk
  | .round index =>
      (schedule.shape.cubeVariables + 1) / Transcript.coinsPerChunk + index.val * count + count
  | .rho index =>
      (schedule.shape.cubeVariables + 1) / Transcript.coinsPerChunk +
        schedule.shape.cubeVariables * count + output + (2 * index.val + 1)

private theorem offset_slot_injective (count output : Nat) (positive : 0 < count)
    {challenge other : schedule.Challenge}
    (same : schedule.offset count output challenge = schedule.offset count output other)
    (sameSlot : challenge.slot = other.slot) : challenge = other := by
  have below (index : Fin schedule.shape.cubeVariables) :
      index.val * count + count ≤ schedule.shape.cubeVariables * count := by
    rw [← Nat.succ_mul]
    exact Nat.mul_le_mul_right _ index.isLt
  have slotValue := congrArg Fin.val sameSlot
  match challenge, other with
  | .alpha coordinate, .alpha coordinate' =>
      simp only [offset, Challenge.slot, slotOf, Transcript.coinsPerChunk_eq] at same slotValue
      exact congrArg Challenge.alpha (Fin.ext (by omega))
  | .alpha coordinate, .gamma =>
      have inside := coordinate.isLt
      simp only [offset, Challenge.slot, slotOf, Transcript.coinsPerChunk_eq] at same slotValue
      omega
  | .alpha coordinate, .round index =>
      have inside := coordinate.isLt
      simp only [offset, Transcript.coinsPerChunk_eq] at same
      generalize index.val * count = steps at same
      omega
  | .alpha coordinate, .rho _ =>
      have inside := coordinate.isLt
      simp only [offset, Transcript.coinsPerChunk_eq] at same
      generalize schedule.shape.cubeVariables * count = total at same
      omega
  | .gamma, .alpha coordinate =>
      have inside := coordinate.isLt
      simp only [offset, Challenge.slot, slotOf, Transcript.coinsPerChunk_eq] at same slotValue
      omega
  | .gamma, .gamma => rfl
  | .gamma, .round index =>
      simp only [offset, Transcript.coinsPerChunk_eq] at same
      generalize index.val * count = steps at same
      omega
  | .gamma, .rho _ =>
      simp only [offset, Transcript.coinsPerChunk_eq] at same
      generalize schedule.shape.cubeVariables * count = total at same
      omega
  | .round index, .alpha coordinate =>
      have inside := coordinate.isLt
      simp only [offset, Transcript.coinsPerChunk_eq] at same
      generalize index.val * count = steps at same
      omega
  | .round index, .gamma =>
      simp only [offset, Transcript.coinsPerChunk_eq] at same
      generalize index.val * count = steps at same
      omega
  | .round index, .round index' =>
      simp only [offset] at same
      have steps : index.val * count = index'.val * count := by omega
      exact congrArg Challenge.round (Fin.ext (Nat.eq_of_mul_eq_mul_right positive steps))
  | .round index, .rho _ =>
      have bound := below index
      simp only [offset] at same
      generalize index.val * count = steps at same bound
      generalize schedule.shape.cubeVariables * count = total at same bound
      omega
  | .rho _, .alpha coordinate =>
      have inside := coordinate.isLt
      simp only [offset, Transcript.coinsPerChunk_eq] at same
      generalize schedule.shape.cubeVariables * count = total at same
      omega
  | .rho _, .gamma =>
      simp only [offset, Transcript.coinsPerChunk_eq] at same
      generalize schedule.shape.cubeVariables * count = total at same
      omega
  | .rho _, .round index' =>
      have bound := below index'
      simp only [offset] at same
      generalize index'.val * count = steps at same bound
      generalize schedule.shape.cubeVariables * count = total at same bound
      omega
  | .rho scalar, .rho scalar' =>
      simp only [offset] at same
      exact congrArg Challenge.rho (Fin.ext (by omega))

private theorem challengeCalls_length (statement : schedule.Statement)
    (messages : schedule.Messages width) (challenge : schedule.Challenge) :
    (schedule.challengeCalls statement messages challenge).length =
      (schedule.statementCalls statement).length +
        schedule.offset (roundCallCount width)
          (absorbCalls (schedule.outputWords messages.output)).length challenge := by
  cases challenge with
  | alpha coordinate =>
      simp only [challengeCalls, proverCalls, fixedCalls, List.length_append,
        List.length_replicate, offset]
  | gamma =>
      simp only [challengeCalls, proverCalls, fixedCalls, List.length_append,
        List.length_replicate, offset]
  | round index =>
      simp only [challengeCalls, proverCalls, fixedCalls, roundPrefixCalls, preRoundCalls,
        List.length_append, List.length_replicate, List.length_nil, takeRounds_length,
        roundCalls_length, offset]
      omega
  | rho index =>
      have scalars : ((Spec.Folding.Nifs.NonInteractive.PiRlcSampler.ScheduleLaw.queryAt []
          (schedule.rhoIndex index)).val.map widen).length = 2 * index.val + 1 := by
        rw [List.length_map,
          Spec.Folding.Nifs.NonInteractive.PiRlcSampler.ScheduleLaw.queryAt_length]
        simp [rhoIndex]
      simp only [challengeCalls, proverCalls, fixedCalls, outputCalls, roundsCalls,
        preRoundCalls, List.length_append, List.length_replicate, allRounds_length, scalars,
        offset]
      omega

theorem challengeCalls_length_eq (statement statement' : schedule.Statement)
    (messages messages' : schedule.Messages width) (challenge : schedule.Challenge) :
    (schedule.challengeCalls statement messages challenge).length =
      (schedule.challengeCalls statement' messages' challenge).length := by
  have output : (absorbCalls (schedule.outputWords messages.output)).length =
      (absorbCalls (schedule.outputWords messages'.output)).length := by
    rw [absorbCalls_length, absorbCalls_length,
      schedule.outputWords_length messages.output messages'.output]
  rw [challengeCalls_length, challengeCalls_length,
    schedule.statementCalls_length statement statement', output]

theorem challengeAddress_injective (statement : schedule.Statement)
    (messages : schedule.Messages width) {challenge other : schedule.Challenge}
    (same : schedule.challengeAddress statement messages challenge =
      schedule.challengeAddress statement messages other) :
    challenge = other := by
  have lengths := congrArg (fun address => address.1.length) same
  simp only [challengeAddress, challengeCalls_length] at lengths
  exact schedule.offset_slot_injective _ _ (roundCallCount_pos width)
    (Nat.add_left_cancel lengths) (congrArg Prod.snd same)

end Lengths

section Identify

variable (schedule : Schedule) {width : Nat}

theorem statementCalls_identify {statement statement' : schedule.Statement}
    (same : schedule.statementCalls statement = schedule.statementCalls statement') :
    statement = statement' :=
  schedule.statementWords_injective (absorbCalls_injective
    (schedule.statementWords_length statement statement') (List.append_cancel_left same))

private theorem roundCalls_injective {messages messages' : schedule.Messages width}
    {round : Fin schedule.shape.cubeVariables}
    (same : schedule.roundCalls messages round = schedule.roundCalls messages' round) :
    messages.rounds round = messages'.rounds round :=
  SumCheck.Finite.FixedPolynomial.eq_of_coefficients (serializeKs_injective
    (absorbCalls_injective ((roundWords_length _).trans (roundWords_length _).symm) same))

private theorem roundsCalls_length (messages messages' : schedule.Messages width)
    (indices : List (Fin schedule.shape.cubeVariables)) :
    (indices.flatMap (schedule.roundCalls messages)).length =
      (indices.flatMap (schedule.roundCalls messages')).length :=
  flatMap_length_eq _ _ _ fun round _ =>
    (schedule.roundCalls_length messages round).trans
      (schedule.roundCalls_length messages' round).symm

private theorem roundsCalls_identify {messages messages' : schedule.Messages width}
    {indices : List (Fin schedule.shape.cubeVariables)}
    (same : indices.flatMap (schedule.roundCalls messages) =
      indices.flatMap (schedule.roundCalls messages'))
    {round : Fin schedule.shape.cubeVariables} (member : round ∈ indices) :
    messages.rounds round = messages'.rounds round :=
  schedule.roundCalls_injective (flatMap_eq_of_lengths _ _ _
    (fun round _ => (schedule.roundCalls_length messages round).trans
      (schedule.roundCalls_length messages' round).symm) same round member)

private theorem preRoundCalls_identify {statement statement' : schedule.Statement}
    {rest rest' : List Call}
    (same : schedule.preRoundCalls statement ++ rest = schedule.preRoundCalls statement' ++ rest') :
    statement = statement' ∧ rest = rest' := by
  have lengths : (schedule.preRoundCalls statement).length =
      (schedule.preRoundCalls statement').length := by
    simp only [preRoundCalls, List.length_append,
      schedule.statementCalls_length statement statement']
  obtain ⟨preRoundEqual, restEqual⟩ := List.append_inj same lengths
  exact ⟨schedule.statementCalls_identify (List.append_inj preRoundEqual
    (schedule.statementCalls_length statement statement')).1, restEqual⟩

private theorem roundPrefixCalls_identify {statement statement' : schedule.Statement}
    {messages messages' : schedule.Messages width} {round : Fin schedule.shape.cubeVariables}
    (same : schedule.roundPrefixCalls statement messages round =
      schedule.roundPrefixCalls statement' messages' round) :
    statement = statement' ∧ ∀ earlier : Fin schedule.shape.cubeVariables,
      earlier.val ≤ round.val → messages.rounds earlier = messages'.rounds earlier := by
  simp only [roundPrefixCalls, List.append_assoc] at same
  obtain ⟨statementEqual, afterPreRound⟩ := schedule.preRoundCalls_identify same
  obtain ⟨earlierEqual, ownEqual⟩ :=
    List.append_inj afterPreRound (schedule.roundsCalls_length messages messages' _)
  refine ⟨statementEqual, fun earlier atMost => ?_⟩
  rcases Nat.lt_or_eq_of_le atMost with before | equal
  · exact schedule.roundsCalls_identify earlierEqual (mem_take_canonicalFinIndices earlier before)
  · rw [Fin.ext equal]
    exact schedule.roundCalls_injective ownEqual

private theorem outputCalls_identify {statement statement' : schedule.Statement}
    {messages messages' : schedule.Messages width}
    (same : schedule.outputCalls statement messages = schedule.outputCalls statement' messages') :
    statement = statement' ∧ messages = messages' := by
  simp only [outputCalls, roundsCalls, List.append_assoc] at same
  obtain ⟨statementEqual, afterPreRound⟩ := schedule.preRoundCalls_identify same
  obtain ⟨roundsEqual, outputEqual⟩ :=
    List.append_inj afterPreRound (schedule.roundsCalls_length messages messages' _)
  have rounds : messages.rounds = messages'.rounds := funext fun round =>
    schedule.roundsCalls_identify roundsEqual (mem_canonicalFinIndices round)
  have output : messages.output = messages'.output :=
    schedule.outputWords_injective
      (absorbCalls_injective (schedule.outputWords_length _ _) outputEqual)
  refine ⟨statementEqual, ?_⟩
  cases messages
  cases messages'
  simp only at rounds output
  subst rounds output
  rfl

theorem AgreeOnAbsorbed.statement_eq {statement statement' : schedule.Statement}
    {messages messages' : schedule.Messages width} {challenge : schedule.Challenge}
    (agree : schedule.AgreeOnAbsorbed statement statement' messages messages' challenge) :
    statement = statement' := by
  cases challenge with
  | alpha => exact agree
  | gamma => exact agree
  | round => exact agree.1
  | rho => exact agree.1

theorem proverCalls_identify {statement statement' : schedule.Statement}
    {messages messages' : schedule.Messages width} (challenge : schedule.Challenge)
    (same : schedule.proverCalls statement messages challenge =
      schedule.proverCalls statement' messages' challenge) :
    schedule.AgreeOnAbsorbed statement statement' messages messages' challenge := by
  cases challenge with
  | alpha => exact schedule.statementCalls_identify same
  | gamma => exact schedule.statementCalls_identify same
  | round => exact schedule.roundPrefixCalls_identify same
  | rho => exact schedule.outputCalls_identify same

end Identify

section Separation

variable (schedule : Schedule) {width : Nat}

theorem challengeCalls_head? (statement : schedule.Statement) (messages : schedule.Messages width)
    (challenge : schedule.Challenge) :
    (schedule.challengeCalls statement messages challenge).head? =
      some (padChunk schedule.domainChunk) := by
  have statementHead : (schedule.statementCalls statement).head? =
      some (padChunk schedule.domainChunk) := by
    have nonempty : schedule.domainChunk ≠ [] := List.ne_nil_of_length_pos (by
      rw [schedule.domainChunk_length]
      decide)
    rw [statementCalls, List.head?_append, absorbCalls_head? nonempty,
      List.take_of_length_le (Nat.le_of_eq schedule.domainChunk_length)]
    rfl
  cases challenge <;>
    simp only [challengeCalls, proverCalls, roundPrefixCalls, outputCalls, roundsCalls,
      preRoundCalls, List.append_assoc, List.head?_append, statementHead, Option.some_or]

private theorem padChunk_injective {left right : List F} (leftLength : left.length = Poseidon2.rate)
    (rightLength : right.length = Poseidon2.rate) (same : padChunk left = padChunk right) :
    left = right := by
  apply List.ext_getElem (leftLength.trans rightLength.symm)
  intro position leftBound rightBound
  have lane := congrFun same ⟨position, leftLength ▸ leftBound⟩
  simpa [padChunk, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem leftBound,
    List.getElem?_eq_getElem rightBound] using lane

/-- Two transcripts with distinct domain chunks share no sponge call prefix: no state of one
transcript is a state of the other. -/
theorem challengeCalls_separated {other : Schedule}
    (distinct : schedule.domainChunk ≠ other.domainChunk) {width' : Nat}
    {statement : schedule.Statement} {messages : schedule.Messages width}
    {challenge : schedule.Challenge} {statement' : other.Statement}
    {messages' : other.Messages width'} {challenge' : other.Challenge} {calls : List Call}
    (left : calls <+: schedule.challengeCalls statement messages challenge)
    (right : calls <+: other.challengeCalls statement' messages' challenge') : calls = [] := by
  cases calls with
  | nil => rfl
  | cons call rest =>
      exfalso
      obtain ⟨leftTail, leftEq⟩ := left
      obtain ⟨rightTail, rightEq⟩ := right
      have leftHead := schedule.challengeCalls_head? statement messages challenge
      have rightHead := other.challengeCalls_head? statement' messages' challenge'
      rw [← leftEq] at leftHead
      rw [← rightEq] at rightHead
      exact distinct (padChunk_injective schedule.domainChunk_length other.domainChunk_length
        (Option.some_injective _ (leftHead.symm.trans rightHead)))

end Separation

section Sponge

def coinPosition (shape : Shape) : FiatShamir.ChallengeLabel shape → Nat
  | .alpha coordinate => coordinate.val
  | .gamma => shape.cubeVariables
  | .sumcheck _ => 0

/-- `Transcript.piCcsOracle` on any shape. -/
def spongeOracle (shape : Shape) :
    Spec.Folding.PiCCS.TranscriptReplay.Oracle K Transcript.State shape where
  transcript :=
    { initialState := fun statement => statement.priorState
      absorbRound := fun state _round message =>
        Transcript.absorb state (Transcript.serializeMessage message)
      squeeze := fun state label => Transcript.squeezeAt state (coinPosition shape label) }
  initialState_is_prior := fun _ => rfl

/-- The deployed sponge at an address: the coin in the slot of the state after the calls. -/
def spongeRead (address : List Call × Slot) : K :=
  readK address.2 (run Transcript.initialState address.1)

private theorem squeeze_eq {shape : Shape} (state : Transcript.State)
    (label : FiatShamir.ChallengeLabel shape) :
    (spongeOracle shape).transcript.squeeze state label =
      (readK (slotOf (coinPosition shape label)) state,
        run state (refresh (coinPosition shape label))) :=
  squeezeAt_eq state (coinPosition shape label)

private theorem squeezeMany_run {shape : Shape} (state : Transcript.State) (start : Nat)
    (labels : List (FiatShamir.ChallengeLabel shape))
    (positions : ∀ index (inside : index < labels.length),
      coinPosition shape labels[index] = start + index) :
    (FiatShamir.squeezeMany (spongeOracle shape).transcript
        (run state (List.replicate (start / Transcript.coinsPerChunk) squeeze)) labels).2 =
      run state (List.replicate ((start + labels.length) / Transcript.coinsPerChunk) squeeze) ∧
    ∀ index, index < labels.length →
      (FiatShamir.squeezeMany (spongeOracle shape).transcript
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
      simp only [FiatShamir.squeezeMany, squeeze_eq, first, next]
      refine ⟨?_, fun index inside => ?_⟩
      · rw [rest.1, List.length_cons, Nat.add_assoc, Nat.add_comm 1]
      · cases index with
        | zero => simp
        | succ index =>
            rw [List.getD_cons_succ, rest.2 index (by simpa using inside),
              Nat.add_assoc, Nat.add_comm 1]

private theorem round_squeeze_eq {shape : Shape} (state : Transcript.State)
    (round : Fin shape.cubeVariables) :
    (spongeOracle shape).transcript.squeeze state (.sumcheck round) =
      (readK (slotOf 0) state, state) := by
  rw [squeeze_eq]
  rfl

private theorem alphaLabels_positions (shape : Shape) (index : Nat)
    (inside : index < (FiatShamir.alphaLabels shape).length) :
    coinPosition shape (FiatShamir.alphaLabels shape)[index] = 0 + index := by
  simp [FiatShamir.alphaLabels, canonicalFinIndices, coinPosition]

variable (schedule : Schedule) {width : Nat}

private theorem absorbRound_eq (messages : schedule.Messages width) (state : Transcript.State)
    (round : Fin schedule.shape.cubeVariables) :
    (spongeOracle schedule.shape).transcript.absorbRound state round
        (messages.certificate.rounds round) =
      run state (schedule.roundCalls messages round) :=
  absorb_eq_run state _

private theorem deriveRoundsFrom_state (messages : schedule.Messages width)
    (state : Transcript.State) (indices : List (Fin schedule.shape.cubeVariables)) :
    (FiatShamir.deriveRoundsFrom (spongeOracle schedule.shape).transcript
        messages.certificate.rounds state indices).2 =
      run state (indices.flatMap (schedule.roundCalls messages)) := by
  induction indices generalizing state with
  | nil => rfl
  | cons round indices inductionHypothesis =>
      simp only [FiatShamir.deriveRoundsFrom, round_squeeze_eq, List.flatMap_cons,
        run_append, absorbRound_eq]
      rw [inductionHypothesis]

private theorem deriveRoundsFrom_getD (messages : schedule.Messages width)
    (state : Transcript.State) (indices : List (Fin schedule.shape.cubeVariables))
    (index : Nat) (bound : index < indices.length) :
    (FiatShamir.deriveRoundsFrom (spongeOracle schedule.shape).transcript
        messages.certificate.rounds state indices).1.getD index K.zero =
      readK (slotOf 0) (run state ((indices.take index).flatMap (schedule.roundCalls messages) ++
        schedule.roundCalls messages indices[index])) := by
  induction indices generalizing state index with
  | nil => simp at bound
  | cons round indices inductionHypothesis =>
      cases index with
      | zero =>
          simp only [FiatShamir.deriveRoundsFrom, round_squeeze_eq, List.getD_cons_zero,
            List.take_zero, List.flatMap_nil, List.nil_append, List.getElem_cons_zero,
            absorbRound_eq]
      | succ index =>
          simp only [FiatShamir.deriveRoundsFrom, round_squeeze_eq, List.getD_cons_succ,
            List.take_succ_cons, List.flatMap_cons, List.getElem_cons_succ]
          rw [inductionHypothesis _ index (by simpa using bound)]
          simp only [run_append, List.append_assoc, absorbRound_eq]

variable (statement : schedule.Statement) (messages : schedule.Messages width)
  {context : Spec.Folding.PiCCS.TranscriptReplay.Statement K Transcript.State schedule.shape}

private theorem alphaSqueezes
    (prior : context.priorState = run Transcript.initialState (schedule.statementCalls statement)) :
    FiatShamir.squeezeMany (spongeOracle schedule.shape).transcript
        ((spongeOracle schedule.shape).transcript.initialState context)
        (FiatShamir.alphaLabels schedule.shape) =
      FiatShamir.squeezeMany (spongeOracle schedule.shape).transcript
        (run (run Transcript.initialState (schedule.statementCalls statement))
          (List.replicate (0 / Transcript.coinsPerChunk) squeeze))
        (FiatShamir.alphaLabels schedule.shape) := by
  rw [Nat.zero_div, List.replicate_zero, run_nil, ← prior]
  rfl

private theorem alpha_read
    (prior : context.priorState = run Transcript.initialState (schedule.statementCalls statement))
    (coordinate : Fin schedule.shape.cubeVariables) :
    (FiatShamir.derive (spongeOracle schedule.shape).transcript context
        messages.certificate).alpha.coordinates.getD coordinate.val K.zero =
      readK (slotOf coordinate.val) (run Transcript.initialState
        (schedule.challengeCalls statement messages (.alpha coordinate))) := by
  change (FiatShamir.squeezeMany (spongeOracle schedule.shape).transcript
      ((spongeOracle schedule.shape).transcript.initialState context)
      (FiatShamir.alphaLabels schedule.shape)).1.getD coordinate.val K.zero = _
  rw [schedule.alphaSqueezes statement prior,
    (squeezeMany_run _ 0 _ (alphaLabels_positions schedule.shape)).2 coordinate.val
      (by simp [FiatShamir.alphaLabels_length]), ← run_append]
  simp only [Nat.zero_add, challengeCalls, proverCalls, fixedCalls]

private theorem alphaState_eq_run
    (prior : context.priorState = run Transcript.initialState (schedule.statementCalls statement)) :
    (FiatShamir.squeezeMany (spongeOracle schedule.shape).transcript
        ((spongeOracle schedule.shape).transcript.initialState context)
        (FiatShamir.alphaLabels schedule.shape)).2 =
      run Transcript.initialState
        (schedule.statementCalls statement ++ schedule.fixedCalls .gamma) := by
  rw [schedule.alphaSqueezes statement prior,
    (squeezeMany_run _ 0 _ (alphaLabels_positions schedule.shape)).1, ← run_append]
  simp only [Nat.zero_add, FiatShamir.alphaLabels_length, fixedCalls]

private theorem gamma_read
    (prior : context.priorState = run Transcript.initialState (schedule.statementCalls statement)) :
    (FiatShamir.derive (spongeOracle schedule.shape).transcript context
        messages.certificate).gamma =
      readK (slotOf schedule.shape.cubeVariables)
        (run Transcript.initialState (schedule.challengeCalls statement messages .gamma)) := by
  change ((spongeOracle schedule.shape).transcript.squeeze
      (FiatShamir.squeezeMany (spongeOracle schedule.shape).transcript
        ((spongeOracle schedule.shape).transcript.initialState context)
        (FiatShamir.alphaLabels schedule.shape)).2 .gamma).1 = _
  rw [squeeze_eq, schedule.alphaState_eq_run statement prior]
  rfl

private theorem preRoundState_eq_run
    (prior : context.priorState = run Transcript.initialState (schedule.statementCalls statement)) :
    (FiatShamir.derivePreSumcheck (spongeOracle schedule.shape).transcript context).state =
      run Transcript.initialState (schedule.preRoundCalls statement) := by
  change ((spongeOracle schedule.shape).transcript.squeeze
      (FiatShamir.squeezeMany (spongeOracle schedule.shape).transcript
        ((spongeOracle schedule.shape).transcript.initialState context)
        (FiatShamir.alphaLabels schedule.shape)).2 .gamma).2 = _
  rw [squeeze_eq, schedule.alphaState_eq_run statement prior, ← run_append, preRoundCalls,
    List.append_assoc]
  simp only [fixedCalls, coinPosition, replicate_refresh]

private theorem round_read
    (prior : context.priorState = run Transcript.initialState (schedule.statementCalls statement))
    (round : Fin schedule.shape.cubeVariables) :
    (FiatShamir.derive (spongeOracle schedule.shape).transcript context
        messages.certificate).roundPoint.coordinates.getD round.val K.zero =
      readK (slotOf 0) (run Transcript.initialState
        (schedule.challengeCalls statement messages (.round round))) := by
  change (FiatShamir.deriveRoundsFrom (spongeOracle schedule.shape).transcript
      messages.certificate.rounds
      (FiatShamir.derivePreSumcheck (spongeOracle schedule.shape).transcript context).state
      (canonicalFinIndices schedule.shape.cubeVariables)).1.getD round.val K.zero = _
  rw [schedule.deriveRoundsFrom_getD _ _ _ _ (by simp [canonicalFinIndices]),
    schedule.preRoundState_eq_run statement prior, ← run_append, canonicalFinIndices_getElem]
  simp only [challengeCalls, proverCalls, roundPrefixCalls, fixedCalls, Fin.eta,
    List.append_assoc, List.append_nil]

private theorem finalState_eq_run
    (prior : context.priorState = run Transcript.initialState (schedule.statementCalls statement)) :
    (FiatShamir.derive (spongeOracle schedule.shape).transcript context
        messages.certificate).finalState =
      run Transcript.initialState (schedule.roundsCalls statement messages) := by
  change (FiatShamir.deriveRoundsFrom (spongeOracle schedule.shape).transcript
      messages.certificate.rounds
      (FiatShamir.derivePreSumcheck (spongeOracle schedule.shape).transcript context).state
      (canonicalFinIndices schedule.shape.cubeVariables)).2 = _
  rw [schedule.deriveRoundsFrom_state, schedule.preRoundState_eq_run statement prior,
    ← run_append]
  rfl

/-- SuperNeo v1.2 Section 7.3 on the sponge: each Π_CCS coin is the read of the sponge state
after the challenge's calls. -/
theorem derive_eq_reads
    (prior : context.priorState = run Transcript.initialState (schedule.statementCalls statement)) :
    FiatShamir.derive (spongeOracle schedule.shape).transcript context messages.certificate =
      { alpha := ⟨List.ofFn fun coordinate => readK (slotOf coordinate.val)
            (run Transcript.initialState
              (schedule.challengeCalls statement messages (.alpha coordinate))),
          by simp⟩
        gamma := readK (slotOf schedule.shape.cubeVariables)
          (run Transcript.initialState (schedule.challengeCalls statement messages .gamma))
        roundPoint := ⟨List.ofFn fun index => readK (slotOf 0) (run Transcript.initialState
            (schedule.challengeCalls statement messages (.round index))),
          by simp⟩
        finalState := run Transcript.initialState (schedule.roundsCalls statement messages) } := by
  have alphaSeal := schedule.alpha_read statement messages prior
  have pointSeal := schedule.round_read statement messages prior
  have gammaSeal := schedule.gamma_read statement messages prior
  have finalSeal := schedule.finalState_eq_run statement messages prior
  revert alphaSeal pointSeal gammaSeal finalSeal
  generalize FiatShamir.derive (spongeOracle schedule.shape).transcript context
    messages.certificate = coins
  rcases coins with ⟨⟨alpha, alphaLength⟩, gamma, ⟨point, pointLength⟩, finalState⟩
  intro alphaSeal pointSeal gammaSeal finalSeal
  simp only at alphaSeal pointSeal gammaSeal finalSeal
  have alphaEqual : alpha = List.ofFn fun coordinate => readK (slotOf coordinate.val)
      (run Transcript.initialState
        (schedule.challengeCalls statement messages (.alpha coordinate))) :=
    List.ext_getElem (by simp [alphaLength]) fun index inside _ =>
      (List.getD_eq_getElem _ K.zero inside).symm.trans
        ((alphaSeal ⟨index, alphaLength ▸ inside⟩).trans (by simp))
  have pointEqual : point = List.ofFn fun index => readK (slotOf 0) (run Transcript.initialState
      (schedule.challengeCalls statement messages (.round index))) :=
    List.ext_getElem (by simp [pointLength]) fun index inside _ =>
      (List.getD_eq_getElem _ K.zero inside).symm.trans
        ((pointSeal ⟨index, pointLength ▸ inside⟩).trans (by simp))
  subst alphaEqual pointEqual gammaSeal finalSeal
  rfl

theorem derive_publicCoins
    (prior : context.priorState = run Transcript.initialState (schedule.statementCalls statement)) :
    ({ alpha := (FiatShamir.derive (spongeOracle schedule.shape).transcript context
          messages.certificate).alpha
       gamma := (FiatShamir.derive (spongeOracle schedule.shape).transcript context
          messages.certificate).gamma
       roundPoint := (FiatShamir.derive (spongeOracle schedule.shape).transcript context
          messages.certificate).roundPoint } : StrongReduction.PublicCoins K schedule.shape) =
      schedule.coinsFrom spongeRead statement messages := by
  rw [schedule.derive_eq_reads statement messages prior]
  rfl

theorem absorbOutput_eq_run :
    Transcript.absorb (run Transcript.initialState (schedule.roundsCalls statement messages))
        (schedule.outputWords messages.output) =
      run Transcript.initialState (schedule.outputCalls statement messages) := by
  rw [absorb_eq_run, ← run_append, outputCalls]

end Sponge

section Rho

variable (schedule : Schedule) {width : Nat}

/-- One draw of the production Π_RLC sampler is the read after the draw's sampler calls. -/
theorem sampleRingChallenge_eq_read (state : Transcript.State) (index : Fin schedule.rhoCount) :
    Transcript.PiRlcSampler.sampleRingChallenge state index.val =
      readRho (run state (schedule.fixedCalls (.rho index))) := by
  rw [fixedCalls, run_map_widen]
  have read := Spec.Folding.Nifs.NonInteractive.PiRlcSampler.TranscriptHistory.queryAt_answer
    state [] (schedule.rhoIndex index)
  rewrite [Transcript.PiRlcSampler.sampleRingChallenge,
    Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.challengeAt,
    Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.scalarAt, readRho,
    ← Spec.Folding.Nifs.NonInteractive.PiRlcSampler.TranscriptHistory.answer, read]
  simp only [Spec.Folding.Nifs.NonInteractive.PiRlcSampler.TranscriptHistory.replay,
    List.foldl_nil, rhoIndex]

/-- SuperNeo v1.2 Section 7.4 on the sponge: the Π_RLC challenge `index` after the Π_CCS output
is the read at the challenge's address. -/
theorem rho_read (statement : schedule.Statement) (messages : schedule.Messages width)
    (index : Fin schedule.rhoCount) :
    Transcript.PiRlcSampler.sampleRingChallenge
        (run Transcript.initialState (schedule.outputCalls statement messages)) index.val =
      readRho (run Transcript.initialState
        (schedule.challengeCalls statement messages (.rho index))) := by
  rw [challengeCalls, proverCalls, run_append]
  exact schedule.sampleRingChallenge_eq_read _ index

end Rho

section OutputWords

/-- One evaluation family on any shape, in the word order of `serializeEvaluations`. -/
def evaluationWords {shape : Shape} (pad : Fin shape.coefficientCount → K)
    (matrix : Fin shape.matrixCount → Fin shape.coefficientCount → K) : List F :=
  (List.finRange shape.coefficientCount).flatMap
      (fun coefficient => serializeK (pad coefficient)) ++
    (List.finRange shape.matrixCount).flatMap fun row =>
      (List.finRange shape.coefficientCount).flatMap fun coefficient =>
        serializeK (matrix row coefficient)

/-- A Π_CCS output on any shape, in the word order of `ProductionKey.fullOutputWords`. -/
def fullOutputWords {shape : Shape} (output : FullOutputCoordinates.FullOutput K shape) :
    List F :=
  (List.finRange shape.sourceCount).flatMap fun source =>
    evaluationWords (output.padCoordinate source) (output.matrixCoordinate source)

private theorem serializeKs_finRange_injective {count : Nat} {left right : Fin count → K}
    (same : (List.finRange count).flatMap (fun index => serializeK (left index)) =
      (List.finRange count).flatMap (fun index => serializeK (right index))) :
    left = right :=
  funext fun index => serializeK_injective (flatMap_eq_of_lengths _
    (fun index => serializeK (left index)) (fun index => serializeK (right index))
    (fun _ _ => rfl) same index (List.mem_finRange index))

private theorem evaluationWords_length {shape : Shape} (pad pad' : Fin shape.coefficientCount → K)
    (matrix matrix' : Fin shape.matrixCount → Fin shape.coefficientCount → K) :
    (evaluationWords pad matrix).length = (evaluationWords pad' matrix').length := by
  simp [evaluationWords, serializeK]

private theorem evaluationWords_injective {shape : Shape}
    {pad pad' : Fin shape.coefficientCount → K}
    {matrix matrix' : Fin shape.matrixCount → Fin shape.coefficientCount → K}
    (same : evaluationWords pad matrix = evaluationWords pad' matrix') :
    pad = pad' ∧ matrix = matrix' := by
  obtain ⟨padWords, matrixWords⟩ := List.append_inj same (by simp [serializeK])
  exact ⟨serializeKs_finRange_injective padWords, funext fun row =>
    serializeKs_finRange_injective (flatMap_eq_of_lengths _ _ _ (fun _ _ => by simp [serializeK])
      matrixWords row (List.mem_finRange row))⟩

theorem fullOutputWords_length {shape : Shape}
    (left right : FullOutputCoordinates.FullOutput K shape) :
    (fullOutputWords left).length = (fullOutputWords right).length :=
  flatMap_length_eq _ _ _ fun _ _ => evaluationWords_length _ _ _ _

theorem fullOutputWords_injective {shape : Shape}
    {left right : FullOutputCoordinates.FullOutput K shape}
    (same : fullOutputWords left = fullOutputWords right) : left = right := by
  have sources := fun source => evaluationWords_injective (flatMap_eq_of_lengths _ _ _
    (fun _ _ => evaluationWords_length _ _ _ _) same source (List.mem_finRange source))
  have pad : left.padCoordinate = right.padCoordinate := funext fun source => (sources source).1
  have matrix : left.matrixCoordinate = right.matrixCoordinate :=
    funext fun source => (sources source).2
  cases left
  cases right
  simp only at pad matrix
  rw [pad, matrix]

end OutputWords

end Schedule

end NightstreamFPrime.Lifecycle.RandomOracle

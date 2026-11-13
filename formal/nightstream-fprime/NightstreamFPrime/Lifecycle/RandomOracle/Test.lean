import Mathlib.Algebra.Order.Archimedean.Real.Basic
import Mathlib.Data.Set.Finite.List
import NightstreamFPrime.Lifecycle.RandomOracle.Schedule
import NightstreamFPrime.Spec.RandomOracle
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.IndependentExecution
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.RoundByRound

/-! The Π_CCS test error in the random-oracle model for any fold transcript `Schedule`: the test
step of the random-oracle chain, which `RandomOracleTest` states for the one-circuit fold. A random
function of each challenge's address (`point`) replaces the Poseidon2 sponge read, so the
Π_CCS coins are `coins oracle statement messages`.

The fold supplies the claimed protocol data of each statement (`source`). The bad set of a
challenge (`bad`) is the `RoundByRound` set of its coin: an `α` coordinate, `γ`, or one round
challenge. A `ρ` challenge has no bad set. Each bad set is local (`bad_local`) and has mass at
most `error` (`bad_mass_le`). When the fold's split lemma (`RoundByRound.falseAcceptance_splits`,
`Light.PiCCS.falseAcceptance_splits`) holds on the oracle coins (`Splits`), one challenge hits
its bad set at its address (`splits_hits`). So an event that implies a hit has probability at
most `(Q + 1) · testError` (`test_error_le`). Inside a game that also queries every challenge
point (`checked`), any local per-challenge family of the same masses gives
`(Q + 2 · cube + 1 + rhoCount) · testError` (`checked_error_le`, `challenges_length`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.RandomOracle.Test

open scoped BigOperators
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.RandomOracle
open NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle.RandomOracle.Sponge
open StrongReduction ConcreteCarrier

attribute [local instance low] Classical.propDecidable

section Answers

/-- One oracle answer: one draw of the production Π_RLC sampler. -/
abbrev Answer := Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Draw

def decodeK (answer : Answer) : K := ⟨answer ⟨0, by decide⟩, answer ⟨1, by decide⟩⟩

private def splitAnswer : Answer ≃ (F × F) × (F × F) where
  toFun answer := ((answer ⟨0, by decide⟩, answer ⟨1, by decide⟩),
    (answer ⟨2, by decide⟩, answer ⟨3, by decide⟩))
  invFun words := fun lane =>
    match lane with
    | ⟨0, _⟩ => words.1.1
    | ⟨1, _⟩ => words.1.2
    | ⟨2, _⟩ => words.2.1
    | _ => words.2.2
  left_inv answer := by
    funext lane
    match lane with
    | ⟨0, _⟩ => rfl
    | ⟨1, _⟩ => rfl
    | ⟨2, _⟩ => rfl
    | ⟨3, _⟩ => rfl
    | ⟨index + 4, inside⟩ =>
        exact absurd inside
          (by unfold Spec.Folding.Nifs.NonInteractive.PiRlcSampler.drawWidth; omega)
  right_inv _ := rfl

/-- Every element of `K`, as the pairs of its two words. Noncomputable: compiled code would build
the `fieldModulus ^ 2` pairs when an executable that imports this module starts. -/
noncomputable def pairs : Finset K :=
  Finset.univ.map ⟨fun words : F × F => (⟨words.1, words.2⟩ : K), fun left right same => by
    cases left
    cases right
    cases same
    rfl⟩

theorem pairs_card : (pairs.card : ℝ) = (fieldModulus ^ 2 : Nat) := by
  simp [pairs, Fintype.card_prod, Fintype.card_fin, pow_two]

/-- A uniform answer decodes to a uniform element of `K`. -/
theorem expect_decodeK (value : K → ℝ) :
    𝔼 answer : Answer, value (decodeK answer) = 𝔼 element ∈ pairs, value element := by
  refine (Fintype.expect_equiv splitAnswer (fun answer => value (decodeK answer))
    (fun words : (F × F) × (F × F) => value ⟨words.1.1, words.1.2⟩) fun _ => rfl).trans ?_
  simp only [pairs, Finset.expect_eq_sum_div_card, Finset.sum_map, Finset.card_map,
    Finset.card_univ, Fintype.card_prod, Fintype.sum_prod_type, Finset.sum_const, nsmul_eq_mul,
    ← Finset.mul_sum]
  have positive : (0 : ℝ) < Fintype.card F := by exact_mod_cast Fintype.card_pos
  push_cast
  field_simp
  rfl

end Answers

section Points

variable (schedule : Schedule) (width : Nat)

private theorem exists_bound : ∃ bound : Nat, ∀ (statement : schedule.Statement)
    (messages : schedule.Messages width) (challenge : schedule.Challenge),
    (schedule.challengeCalls statement messages challenge).length ≤ bound := by
  by_cases inhabited : Nonempty (schedule.Statement × schedule.Messages width)
  · obtain ⟨statement₀, messages₀⟩ := inhabited
    refine ⟨∑ challenge, (schedule.challengeCalls statement₀ messages₀ challenge).length,
      fun statement messages challenge => ?_⟩
    rw [schedule.challengeCalls_length_eq statement statement₀ messages messages₀]
    exact Finset.single_le_sum
      (f := fun challenge => (schedule.challengeCalls statement₀ messages₀ challenge).length)
      (fun _ _ => Nat.zero_le _) (Finset.mem_univ challenge)
  · exact ⟨0, fun statement messages _ => absurd ⟨⟨statement, messages⟩⟩ inhabited⟩

noncomputable def pointBound : Nat :=
  Classical.choose (exists_bound schedule width)

abbrev BoundedCalls := {calls : List Call // calls.length ≤ pointBound schedule width}

noncomputable instance : Fintype (BoundedCalls schedule width) :=
  (List.finite_length_le Call _).fintype

/-- A random-oracle point: the sponge calls before a read and the lane pair of the read. -/
abbrev Point := BoundedCalls schedule width × Slot

variable {schedule width}

noncomputable def point (statement : schedule.Statement) (messages : schedule.Messages width)
    (challenge : schedule.Challenge) : Point schedule width :=
  (⟨schedule.challengeCalls statement messages challenge,
    Classical.choose_spec (exists_bound schedule width) statement messages challenge⟩,
    challenge.slot)

theorem point_injective (statement : schedule.Statement) (messages : schedule.Messages width)
    {challenge other : schedule.Challenge}
    (same : point statement messages challenge = point statement messages other) :
    challenge = other :=
  schedule.challengeAddress_injective statement messages (by
    have calls := congrArg (fun target => target.1.val) same
    have slot := congrArg Prod.snd same
    exact Prod.ext calls slot)

noncomputable def read (oracle : Point schedule width → Answer) (address : List Call × Slot) :
    K :=
  if fits : address.1.length ≤ pointBound schedule width then
    decodeK (oracle (⟨address.1, fits⟩, address.2))
  else K.zero

/-- The Π_CCS coins that the transcript reads from `oracle`. -/
noncomputable def coins (oracle : Point schedule width → Answer) (statement : schedule.Statement)
    (messages : schedule.Messages width) : PublicCoins K schedule.shape :=
  schedule.coinsFrom (read oracle) statement messages

theorem read_challengeAddress (oracle : Point schedule width → Answer)
    (statement : schedule.Statement) (messages : schedule.Messages width)
    (challenge : schedule.Challenge) :
    read oracle (schedule.challengeAddress statement messages challenge) =
      decodeK (oracle (point statement messages challenge)) :=
  dite_eq_left _

end Points

section Bad

variable {schedule : Schedule} {width : Nat}

/-- SuperNeo v1.2 Appendix B.2 (`RoundByRound`): the bad set of one challenge's coin on the
claimed protocol data `data`. -/
def coinBad (data : ProtocolPolynomial.Data K schedule.shape) (challenge : schedule.Challenge)
    (coins : PublicCoins K schedule.shape) (messages : schedule.Messages width) : Set K :=
  match challenge with
  | .alpha coordinate =>
      RoundByRound.alphaBad (data.toJointData extensionOps) coins.alpha coordinate
  | .gamma => RoundByRound.gammaBad (data.toJointData extensionOps) coins.alpha
  | .round index => RoundByRound.roundBad data coins.alpha coins.gamma
      (coins.roundPoint.coordinates.take index.val) (messages.rounds index)
      (schedule.shape.cubeVariables - index.val - 1)
  | .rho _ => ∅

/-- The answers at `target` that put `challenge` in its bad set, for any statement and messages
whose `challenge` address is `target`. -/
def bad (source : schedule.Statement → ProtocolPolynomial.Data K schedule.shape)
    (challenge : schedule.Challenge) (target : Point schedule width)
    (oracle : Point schedule width → Answer) : Set Answer :=
  {answer | ∃ (statement : schedule.Statement) (messages : schedule.Messages width),
    point statement messages challenge = target ∧
      decodeK answer ∈ coinBad (source statement) challenge (coins oracle statement messages)
        messages}

variable (schedule width) in
/-- The mass bound of each challenge's bad set: `1/q²` for an `α` coordinate, `(J − 1)/q²` for
`γ`, `width/q²` for a round, and none for `ρ`. -/
noncomputable def error : schedule.Challenge → ℝ
  | .alpha _ => 1 / (fieldModulus ^ 2 : Nat)
  | .gamma => (schedule.shape.jointCoefficientCount - 1 : Nat) / (fieldModulus ^ 2 : Nat)
  | .round _ => width / (fieldModulus ^ 2 : Nat)
  | .rho _ => 0

theorem error_nonnegative (challenge : schedule.Challenge) :
    0 ≤ error schedule width challenge := by
  cases challenge <;> simp only [error] <;> positivity

private theorem cubePoint_eq {size : Nat} {left right : CubePoint K size}
    (same : left.coordinates = right.coordinates) : left = right := by
  cases left
  cases right
  cases same
  rfl

private theorem read_update (oracle : Point schedule width → Answer)
    (statement : schedule.Statement) (messages : schedule.Messages width)
    {challenge other : schedule.Challenge} (different : other ≠ challenge) (answer : Answer) :
    read (Function.update oracle (point statement messages challenge) answer)
        (schedule.challengeAddress statement messages other) =
      read oracle (schedule.challengeAddress statement messages other) := by
  have apart : point statement messages other ≠ point statement messages challenge := fun same =>
    different (point_injective statement messages same)
  rw [read_challengeAddress, read_challengeAddress, Function.update_of_ne apart]

private theorem alpha_update (oracle : Point schedule width → Answer)
    (statement : schedule.Statement) (messages : schedule.Messages width)
    {challenge : schedule.Challenge} (notAlpha : ∀ index, challenge ≠ .alpha index)
    (answer : Answer) :
    (coins (Function.update oracle (point statement messages challenge) answer) statement
        messages).alpha = (coins oracle statement messages).alpha := by
  apply cubePoint_eq
  simp only [coins, Schedule.coinsFrom]
  congr 1
  funext index
  exact read_update oracle statement messages (fun same => notAlpha index same.symm) answer

private theorem alpha_update_self (oracle : Point schedule width → Answer)
    (statement : schedule.Statement) (messages : schedule.Messages width)
    (index : Fin schedule.shape.cubeVariables) (answer : Answer) :
    (coins (Function.update oracle (point statement messages (.alpha index)) answer) statement
        messages).alpha =
      ⟨(coins oracle statement messages).alpha.coordinates.set index.val (decodeK answer),
        by simp [(coins oracle statement messages).alpha.dimension]⟩ := by
  apply cubePoint_eq
  simp only [coins, Schedule.coinsFrom]
  apply List.ext_getElem (by simp)
  intro position inside _
  simp only [List.getElem_ofFn]
  by_cases same : position = index.val
  · subst same
    simp [read_challengeAddress]
  · rw [List.getElem_set_ne (Ne.symm same), List.getElem_ofFn]
    exact read_update oracle statement messages
      (fun equal => same (congrArg Fin.val (Schedule.Challenge.alpha.inj equal))) answer

private theorem gamma_update (oracle : Point schedule width → Answer)
    (statement : schedule.Statement) (messages : schedule.Messages width)
    {challenge : schedule.Challenge} (notGamma : challenge ≠ .gamma) (answer : Answer) :
    (coins (Function.update oracle (point statement messages challenge) answer) statement
        messages).gamma = (coins oracle statement messages).gamma :=
  read_update oracle statement messages (Ne.symm notGamma) answer

private theorem rounds_update (oracle : Point schedule width → Answer)
    (statement : schedule.Statement) (messages : schedule.Messages width)
    (index : Fin schedule.shape.cubeVariables) (answer : Answer) :
    (coins (Function.update oracle (point statement messages (.round index)) answer) statement
        messages).roundPoint.coordinates.take index.val =
      (coins oracle statement messages).roundPoint.coordinates.take index.val := by
  simp only [coins, Schedule.coinsFrom]
  apply List.ext_getElem (by simp)
  intro position inside _
  simp only [List.getElem_take, List.getElem_ofFn]
  have below : position < index.val := by simp at inside; omega
  exact read_update oracle statement messages (fun equal => by
    have := congrArg Fin.val (Schedule.Challenge.round.inj equal)
    simp at this
    omega) answer

private theorem coinBad_update (data : ProtocolPolynomial.Data K schedule.shape)
    (challenge : schedule.Challenge) (oracle : Point schedule width → Answer)
    (statement : schedule.Statement) (messages : schedule.Messages width) (answer : Answer) :
    coinBad data challenge
        (coins (Function.update oracle (point statement messages challenge) answer) statement
          messages) messages =
      coinBad data challenge (coins oracle statement messages) messages := by
  cases challenge with
  | alpha index =>
      simp only [coinBad]
      rw [alpha_update_self, RoundByRound.alphaBad_set]
  | gamma =>
      simp only [coinBad]
      rw [alpha_update oracle statement messages (fun _ => by simp)]
  | round index =>
      simp only [coinBad]
      rw [alpha_update oracle statement messages (fun _ => by simp),
        gamma_update oracle statement messages (by simp), rounds_update]
  | rho => rfl

theorem bad_local (source : schedule.Statement → ProtocolPolynomial.Data K schedule.shape)
    (challenge : schedule.Challenge) : Local (bad (width := width) source challenge) := by
  intro target oracle answer
  ext candidate
  constructor
  · rintro ⟨statement, messages, located, inside⟩
    subst located
    exact ⟨statement, messages, rfl, by rwa [coinBad_update] at inside⟩
  · rintro ⟨statement, messages, located, inside⟩
    subst located
    exact ⟨statement, messages, rfl, by rwa [coinBad_update]⟩

private theorem below_of_mem_take {count bound : Nat} {index : Fin count}
    (member : index ∈ (canonicalFinIndices count).take bound) : index.val < bound := by
  obtain ⟨position, inside, located⟩ := List.getElem_of_mem member
  subst located
  simp only [List.length_take, canonicalFinIndices, List.length_ofFn] at inside
  simp only [canonicalFinIndices, List.getElem_take, List.getElem_ofFn, id]
  omega

private theorem challengeCalls_round_congr (statement : schedule.Statement)
    (messages messages' : schedule.Messages width) (index : Fin schedule.shape.cubeVariables)
    (agree : ∀ earlier : Fin schedule.shape.cubeVariables, earlier.val ≤ index.val →
      messages.rounds earlier = messages'.rounds earlier) :
    schedule.challengeCalls statement messages (.round index) =
      schedule.challengeCalls statement messages' (.round index) := by
  have message (earlier : Fin schedule.shape.cubeVariables) (atMost : earlier.val ≤ index.val) :
      schedule.roundCalls messages earlier = schedule.roundCalls messages' earlier := by
    simp only [Schedule.roundCalls, agree earlier atMost]
  have earlierRounds :
      ((canonicalFinIndices schedule.shape.cubeVariables).take index.val).flatMap
          (schedule.roundCalls messages) =
        ((canonicalFinIndices schedule.shape.cubeVariables).take index.val).flatMap
          (schedule.roundCalls messages') := by
    simp only [List.flatMap]
    congr 1
    apply List.map_congr_left
    intro earlier member
    exact message earlier (below_of_mem_take member).le
  simp only [Schedule.challengeCalls, Schedule.proverCalls, Schedule.roundPrefixCalls,
    earlierRounds, message index le_rfl]

private theorem coordinate_alpha (oracle : Point schedule width → Answer)
    (statement : schedule.Statement) (messages : schedule.Messages width)
    (index : Fin schedule.shape.cubeVariables) :
    RoundByRound.coordinate (coins oracle statement messages).alpha index =
      decodeK (oracle (point statement messages (.alpha index))) := by
  simp only [RoundByRound.coordinate, coins, Schedule.coinsFrom, List.getElem_ofFn,
    read_challengeAddress]

private theorem coordinate_round (oracle : Point schedule width → Answer)
    (statement : schedule.Statement) (messages : schedule.Messages width)
    (index : Fin schedule.shape.cubeVariables) :
    RoundByRound.coordinate (coins oracle statement messages).roundPoint index =
      decodeK (oracle (point statement messages (.round index))) := by
  simp only [RoundByRound.coordinate, coins, Schedule.coinsFrom, List.getElem_ofFn,
    read_challengeAddress]

private theorem alpha_gamma_messages_irrelevant (oracle : Point schedule width → Answer)
    (statement : schedule.Statement) (messages messages' : schedule.Messages width) :
    (coins oracle statement messages).alpha = (coins oracle statement messages').alpha ∧
      (coins oracle statement messages).gamma = (coins oracle statement messages').gamma := by
  have alpha (index : Fin schedule.shape.cubeVariables) :
      schedule.challengeAddress statement messages (.alpha index) =
        schedule.challengeAddress statement messages' (.alpha index) :=
    rfl
  have gamma : schedule.challengeAddress statement messages .gamma =
      schedule.challengeAddress statement messages' .gamma := rfl
  refine ⟨cubePoint_eq ?_, ?_⟩
  · simp only [coins, Schedule.coinsFrom, alpha]
  · simp only [coins, Schedule.coinsFrom, gamma]

private theorem coinBad_congr
    (source : schedule.Statement → ProtocolPolynomial.Data K schedule.shape)
    (challenge : schedule.Challenge) (oracle : Point schedule width → Answer)
    {statement statement' : schedule.Statement} {messages messages' : schedule.Messages width}
    (same : point statement messages challenge = point statement' messages' challenge) :
    coinBad (source statement) challenge (coins oracle statement messages) messages =
      coinBad (source statement') challenge (coins oracle statement' messages') messages' := by
  have calls : schedule.challengeCalls statement messages challenge =
      schedule.challengeCalls statement' messages' challenge :=
    congrArg (fun target => target.1.val) same
  have agree := schedule.proverCalls_identify challenge (List.append_cancel_right calls)
  cases challenge with
  | alpha index =>
      have equal : statement = statement' := agree
      subst equal
      simp only [coinBad]
      rw [(alpha_gamma_messages_irrelevant oracle statement messages messages').1]
  | gamma =>
      have equal : statement = statement' := agree
      subst equal
      simp only [coinBad]
      rw [(alpha_gamma_messages_irrelevant oracle statement messages messages').1]
  | round index =>
      obtain ⟨equal, rounds⟩ := agree
      subst equal
      have earlier : (coins oracle statement messages).roundPoint.coordinates.take index.val =
          (coins oracle statement messages').roundPoint.coordinates.take index.val := by
        simp only [coins, Schedule.coinsFrom]
        apply List.ext_getElem (by simp)
        intro position inside _
        have below : position < index.val := by simp at inside; omega
        simp only [List.getElem_take, List.getElem_ofFn, Schedule.challengeAddress]
        rw [challengeCalls_round_congr statement messages messages' ⟨position, by omega⟩
          fun round atMost => rounds round (by simp at atMost; omega)]
      simp only [coinBad]
      rw [earlier, rounds index le_rfl,
        (alpha_gamma_messages_irrelevant oracle statement messages messages').1,
        (alpha_gamma_messages_irrelevant oracle statement messages messages').2]
  | rho => simp only [coinBad]

private theorem coinBad_probability_le (data : ProtocolPolynomial.Data K schedule.shape)
    (degreeCovers : data.toVerifierInput.sumcheckDegreeBound ≤ width)
    (challenge : schedule.Challenge) (coins : PublicCoins K schedule.shape)
    (messages : schedule.Messages width) :
    (𝔼 element ∈ pairs,
      if element ∈ coinBad data challenge coins messages then (1 : ℝ) else 0) ≤
      error schedule width challenge := by
  cases challenge with
  | alpha index =>
      calc
        _ = 𝔼 element ∈ pairs, if element ∈ RoundByRound.alphaBad
              (data.toJointData extensionOps) coins.alpha index then (1 : ℝ) else 0 :=
          Finset.expect_congr rfl fun _ _ => if_congr Iff.rfl rfl rfl
        _ ≤ 1 / (pairs.card : ℝ) :=
          RoundByRound.alphaBad_probability_le pairs _ coins.alpha index
        _ = error schedule width (.alpha index) := by rw [pairs_card]; rfl
  | gamma =>
      calc
        _ = 𝔼 element ∈ pairs, if element ∈ RoundByRound.gammaBad
              (data.toJointData extensionOps) coins.alpha then (1 : ℝ) else 0 :=
          Finset.expect_congr rfl fun _ _ => if_congr Iff.rfl rfl rfl
        _ ≤ (schedule.shape.jointCoefficientCount - 1 : Nat) / (pairs.card : ℝ) :=
          RoundByRound.gammaBad_probability_le pairs _ coins.alpha
        _ = error schedule width .gamma := by rw [pairs_card]; rfl
  | round index =>
      have length : (coins.roundPoint.coordinates.take index.val).length + 1 +
          (schedule.shape.cubeVariables - index.val - 1) = schedule.shape.cubeVariables := by
        have below := index.isLt
        simp only [List.length_take, coins.roundPoint.dimension]
        omega
      calc
        _ = 𝔼 element ∈ pairs,
            if element ∈ RoundByRound.roundBad data coins.alpha coins.gamma
              (coins.roundPoint.coordinates.take index.val) (messages.rounds index)
              (schedule.shape.cubeVariables - index.val - 1) then (1 : ℝ) else 0 :=
          Finset.expect_congr rfl fun _ _ => if_congr Iff.rfl rfl rfl
        _ ≤ width / (pairs.card : ℝ) := RoundByRound.roundBad_probability_le pairs _ coins.alpha
          coins.gamma degreeCovers _ (messages.rounds index) _ length
        _ = error schedule width (.round index) := by rw [pairs_card]; rfl
  | rho =>
      simp only [coinBad, error, Set.mem_empty_iff_false, ite_false, Finset.expect_const_zero,
        le_refl]

theorem bad_mass_le (source : schedule.Statement → ProtocolPolynomial.Data K schedule.shape)
    (degreeCovers : ∀ statement, (source statement).toVerifierInput.sumcheckDegreeBound ≤ width)
    (challenge : schedule.Challenge) (target : Point schedule width)
    (oracle : Point schedule width → Answer) :
    mass (bad source challenge target oracle) ≤ error schedule width challenge := by
  by_cases located : ∃ (statement : schedule.Statement) (messages : schedule.Messages width),
      point statement messages challenge = target
  · obtain ⟨statement₀, messages₀, rfl⟩ := located
    have inside : bad source challenge (point statement₀ messages₀ challenge) oracle ⊆
        {answer | decodeK answer ∈ coinBad (source statement₀) challenge
          (coins oracle statement₀ messages₀) messages₀} := by
      rintro answer ⟨statement, messages, same, member⟩
      rwa [coinBad_congr source challenge oracle same] at member
    refine (mass_mono inside).trans ?_
    have uniform := expect_decodeK (fun element => if element ∈ coinBad (source statement₀)
      challenge (coins oracle statement₀ messages₀) messages₀ then (1 : ℝ) else 0)
    have bound := coinBad_probability_le (source statement₀) (degreeCovers statement₀) challenge
      (coins oracle statement₀ messages₀) messages₀
    unfold mass
    exact (le_of_eq uniform).trans bound
  · have empty : bad source challenge target oracle = ∅ := by
      ext answer
      simp only [bad, Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false]
      rintro ⟨statement, messages, same, _⟩
      exact located ⟨statement, messages, same⟩
    rw [empty]
    simp only [mass, Set.mem_empty_iff_false, ite_false, Finset.expect_const_zero]
    exact error_nonnegative challenge

end Bad

section Hits

variable {schedule : Schedule} {width : Nat}

/-- The conclusion of a fold's Π_CCS split lemma (`RoundByRound.falseAcceptance_splits`,
`Light.PiCCS.falseAcceptance_splits`) on the claimed data `data` and the probe `probe`. -/
def Splits {shape : Shape} (data : ProtocolPolynomial.Data K shape) (width : Nat)
    (probe : Probe K shape) : Prop :=
  (∃ index, RoundByRound.coordinate probe.coins.alpha index ∈
      RoundByRound.alphaBad (data.toJointData extensionOps) probe.coins.alpha index) ∨
    probe.coins.gamma ∈
      RoundByRound.gammaBad (data.toJointData extensionOps) probe.coins.alpha ∨
    ∃ (certificate : FixedPhase.Certificate K width) (round : Fin shape.cubeVariables)
      (message : FixedPolynomial K width),
      FixedPhase.RawCertificate.decode width probe.response.rounds = some certificate ∧
      certificate.rounds[round.val]? = some message ∧
      RoundByRound.coordinate probe.coins.roundPoint round ∈
        RoundByRound.roundBad data probe.coins.alpha probe.coins.gamma
          (probe.coins.roundPoint.coordinates.take round.val) message
          (shape.cubeVariables - round.val - 1)

/-- An `α` coordinate, `γ`, or a round challenge of the transcript at `statement` and
`messages` reads an answer of `family` at its own point. -/
def Hits
    (family : schedule.Challenge → Point schedule width → (Point schedule width → Answer) →
      Set Answer)
    (oracle : Point schedule width → Answer) (statement : schedule.Statement)
    (messages : schedule.Messages width) : Prop :=
  (∃ index, oracle (point statement messages (.alpha index)) ∈
      family (.alpha index) (point statement messages (.alpha index)) oracle) ∨
    oracle (point statement messages .gamma) ∈ family .gamma (point statement messages .gamma)
      oracle ∨
    ∃ index, oracle (point statement messages (.round index)) ∈
      family (.round index) (point statement messages (.round index)) oracle

/-- A split on the oracle coins, with the rounds of `messages` as the probe's certificate, is a
hit of one challenge's bad set at its address. -/
theorem splits_hits (source : schedule.Statement → ProtocolPolynomial.Data K schedule.shape)
    (oracle : Point schedule width → Answer) (statement : schedule.Statement)
    (messages : schedule.Messages width) (probe : Probe K schedule.shape)
    (coinsEq : probe.coins = coins oracle statement messages)
    (roundsEq : probe.response.rounds =
      FixedPhase.RawCertificate.encode ⟨List.ofFn messages.rounds⟩)
    (splits : Splits (source statement) width probe) :
    Hits (bad source) oracle statement messages := by
  unfold Splits at splits
  rw [coinsEq] at splits
  rcases splits with ⟨index, member⟩ | member |
    ⟨certificate, index, message, decoded, located, member⟩
  · refine Or.inl ⟨index, statement, messages, rfl, ?_⟩
    rwa [coordinate_alpha] at member
  · refine Or.inr (Or.inl ⟨statement, messages, rfl, ?_⟩)
    change read oracle (schedule.challengeAddress statement messages .gamma) ∈ _ at member
    rwa [read_challengeAddress] at member
  · refine Or.inr (Or.inr ⟨index, statement, messages, rfl, ?_⟩)
    rw [roundsEq, FixedPhase.RawCertificate.decode_encode] at decoded
    cases decoded
    have equal : message = messages.rounds index := by simpa using located.symm
    subst equal
    rwa [coordinate_round] at member

private theorem indicator_le_sum {Index : Type} [Fintype Index] (event : Index → Prop)
    (chosen : Index) (holds : event chosen) :
    (1 : ℝ) ≤ ∑ index, if event index then (1 : ℝ) else 0 := by
  have single := Finset.single_le_sum (f := fun index => if event index then (1 : ℝ) else 0)
    (fun index _ => by split <;> norm_num) (Finset.mem_univ chosen)
  simpa only [ite_eq_left holds] using single

private theorem indicator_sum_nonnegative {Index : Type} [Fintype Index] (event : Index → Prop) :
    (0 : ℝ) ≤ ∑ index, if event index then (1 : ℝ) else 0 :=
  Finset.sum_nonneg fun index _ => by split <;> norm_num

variable (schedule width) in
/-- SuperNeo v1.2 Appendix B.2, equation (14): an event that implies a hit of an `α`
coordinate, `γ`, or a round challenge has probability at most `scale · testError` when each
challenge's hit has probability at most `scale · error`. -/
theorem hits_error_le {Sample : Type} [Fintype Sample] (event : Sample → Prop)
    (hit : schedule.Challenge → Sample → Prop) (scale : ℝ)
    (hits : ∀ sample, event sample → (∃ index, hit (.alpha index) sample) ∨
      hit .gamma sample ∨ ∃ index, hit (.round index) sample)
    (chance : ∀ challenge, 𝔼 sample, (if hit challenge sample then (1 : ℝ) else 0) ≤
      scale * error schedule width challenge) :
    𝔼 sample, (if event sample then (1 : ℝ) else 0) ≤
      scale * IndependentExecution.testError schedule.shape width := by
  have split (sample : Sample) :
      (if event sample then (1 : ℝ) else 0) ≤
        (∑ index, if hit (.alpha index) sample then (1 : ℝ) else 0) +
          (if hit .gamma sample then (1 : ℝ) else 0) +
          ∑ index, if hit (.round index) sample then (1 : ℝ) else 0 := by
    have alphas := indicator_sum_nonnegative fun index => hit (.alpha index) sample
    have rounds := indicator_sum_nonnegative fun index => hit (.round index) sample
    have gamma : (0 : ℝ) ≤ if hit .gamma sample then (1 : ℝ) else 0 := by split <;> norm_num
    by_cases happens : event sample
    · rw [ite_eq_left happens]
      rcases hits sample happens with ⟨index, holds⟩ | holds | ⟨index, holds⟩
      · have := indicator_le_sum (fun index => hit (.alpha index) sample) index holds
        linarith
      · have : (1 : ℝ) ≤ if hit .gamma sample then (1 : ℝ) else 0 := by
          rw [ite_eq_left holds]
        linarith
      · have := indicator_le_sum (fun index => hit (.round index) sample) index holds
        linarith
    · rw [ite_eq_right happens]
      linarith
  have alphaBound : 𝔼 sample, (∑ index, if hit (.alpha index) sample then (1 : ℝ) else 0) ≤
      ∑ _index : Fin schedule.shape.cubeVariables,
        scale * (1 / (fieldModulus ^ 2 : Nat)) := by
    rw [Finset.expect_sum_comm]
    exact Finset.sum_le_sum fun index _ => chance (.alpha index)
  have roundBound : 𝔼 sample, (∑ index, if hit (.round index) sample then (1 : ℝ) else 0) ≤
      ∑ _index : Fin schedule.shape.cubeVariables,
        scale * (width / (fieldModulus ^ 2 : Nat)) := by
    rw [Finset.expect_sum_comm]
    exact Finset.sum_le_sum fun index _ => chance (.round index)
  have total :
      𝔼 sample, (if event sample then (1 : ℝ) else 0) ≤
        (𝔼 sample, ∑ index, if hit (.alpha index) sample then (1 : ℝ) else 0) +
          (𝔼 sample, if hit .gamma sample then (1 : ℝ) else 0) +
          𝔼 sample, ∑ index, if hit (.round index) sample then (1 : ℝ) else 0 := by
    refine (Finset.expect_le_expect fun sample _ => split sample).trans (le_of_eq ?_)
    rw [Finset.expect_add_distrib, Finset.expect_add_distrib]
  refine total.trans ((add_le_add (add_le_add alphaBound (chance .gamma)) roundBound).trans
    (le_of_eq ?_))
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, error,
    IndependentExecution.testError]
  push_cast
  ring

/-- The test step with the claimed data fixed: an event that implies a hit of the bad sets on
`source` has probability at most `(Q + 1) · testError` over the oracle, for an adversary with
at most `Q` queries. -/
theorem test_error_le (source : schedule.Statement → ProtocolPolynomial.Data K schedule.shape)
    (degreeCovers : ∀ statement, (source statement).toVerifierInput.sumcheckDegreeBound ≤ width)
    {Output : Type} (adversary : OracleComp (Point schedule width) Answer Output)
    {queries : Nat} (bounded : adversary.QueryBound queries)
    (statement : Output → schedule.Statement) (messages : Output → schedule.Messages width)
    (event : (Point schedule width → Answer) → Prop)
    (hits : ∀ oracle, event oracle →
      Hits (bad source) oracle (statement (adversary.run oracle))
        (messages (adversary.run oracle))) :
    𝔼 oracle, (if event oracle then (1 : ℝ) else 0) ≤
      (queries + 1) * IndependentExecution.testError schedule.shape width :=
  hits_error_le schedule width event
    (fun challenge oracle =>
      oracle (point (statement (adversary.run oracle)) (messages (adversary.run oracle))
          challenge) ∈
        bad source challenge
          (point (statement (adversary.run oracle)) (messages (adversary.run oracle)) challenge)
          oracle)
    _ hits
    fun challenge => pinned_le (bad source challenge) (bad_local source challenge)
      (error schedule width challenge) (bad_mass_le source degreeCovers challenge)
      (error_nonnegative challenge) bounded
      (fun output => point (statement output) (messages output) challenge)

end Hits

section Checked

variable (schedule : Schedule)

/-- Every challenge of the transcript, in transcript order. -/
def challenges : List schedule.Challenge :=
  List.ofFn Schedule.Challenge.alpha ++ [Schedule.Challenge.gamma] ++
    List.ofFn Schedule.Challenge.round ++ List.ofFn Schedule.Challenge.rho

theorem mem_challenges (challenge : schedule.Challenge) : challenge ∈ challenges schedule := by
  cases challenge <;> simp [challenges]

theorem challenges_length :
    (challenges schedule).length =
      schedule.shape.cubeVariables + 1 + schedule.shape.cubeVariables + schedule.rhoCount := by
  simp [challenges]
  omega

variable {schedule} {width : Nat} {Output : Type}
  (adversary : OracleComp (Point schedule width) Answer Output)
  (statement : Output → schedule.Statement) (messages : Output → schedule.Messages width)

/-- The adversary, then a query at every challenge point of its output. -/
noncomputable def checked : OracleComp (Point schedule width) Answer Output :=
  adversary.thenQueries fun output =>
    (challenges schedule).map (point (statement output) (messages output))

theorem checked_bound {queries : Nat} (bounded : adversary.QueryBound queries) :
    (checked adversary statement messages).QueryBound
      (queries + (challenges schedule).length) :=
  bounded.thenQueries fun _ => by simp

theorem mem_checked (oracle : Point schedule width → Answer) (challenge : schedule.Challenge) :
    point (statement (adversary.run oracle)) (messages (adversary.run oracle)) challenge ∈
      (checked adversary statement messages).queries oracle :=
  OracleComp.mem_queries_thenQueries _ _ _
    (List.mem_map.mpr ⟨challenge, mem_challenges schedule challenge, rfl⟩)

/-- The test step for any local per-challenge family with the masses of `error`: inside the checked
game, an event that implies a hit has probability at most
`(Q + challenge count) · testError`. -/
theorem checked_error_le
    (family : schedule.Challenge → Point schedule width → (Point schedule width → Answer) →
      Set Answer)
    (local_ : ∀ challenge, Local (family challenge))
    (small : ∀ challenge target oracle,
      mass (family challenge target oracle) ≤ error schedule width challenge)
    {queries : Nat} (bounded : adversary.QueryBound queries)
    (event : (Point schedule width → Answer) → Prop)
    (hits : ∀ oracle, event oracle →
      Hits family oracle (statement (adversary.run oracle)) (messages (adversary.run oracle))) :
    𝔼 oracle, (if event oracle then (1 : ℝ) else 0) ≤
      ((queries + (challenges schedule).length : Nat) : ℝ) *
        IndependentExecution.testError schedule.shape width := by
  refine hits_error_le schedule width event
    (fun challenge oracle =>
      oracle (point (statement (adversary.run oracle)) (messages (adversary.run oracle))
          challenge) ∈
        family challenge
          (point (statement (adversary.run oracle)) (messages (adversary.run oracle)) challenge)
          oracle)
    _ hits fun challenge => ?_
  have escape := escape_le (family challenge) (local_ challenge) (error schedule width challenge)
    (small challenge) (error_nonnegative challenge)
    (checked_bound adversary statement messages bounded)
  refine (Finset.expect_le_expect fun oracle _ => ?_).trans escape
  by_cases isBad :
      oracle (point (statement (adversary.run oracle)) (messages (adversary.run oracle))
          challenge) ∈
        family challenge
          (point (statement (adversary.run oracle)) (messages (adversary.run oracle)) challenge)
          oracle
  · have queried : ∃ asked ∈ (checked adversary statement messages).queries oracle,
        oracle asked ∈ family challenge asked oracle :=
      ⟨_, mem_checked adversary statement messages oracle challenge, isBad⟩
    simp only [ite_eq_left isBad, ite_eq_left queried]
    norm_num
  · simp only [ite_eq_right isBad]
    split <;> norm_num

end Checked

end NightstreamFPrime.Lifecycle.RandomOracle.Test

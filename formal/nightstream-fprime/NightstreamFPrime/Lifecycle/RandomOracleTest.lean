import Mathlib.Data.Set.Finite.List
import NightstreamFPrime.Lifecycle.TranscriptCoverage
import NightstreamFPrime.Spec.RandomOracle
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.IndependentExecution
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.RoundByRound

/-!
Owns the PiCCS test error of the production NIFS key in the random-oracle
model: a random function of each challenge's calls replaces the sponge read.

Inputs: the production key, a fixed running statement, a fixed witness, and an
adaptive oracle adversary with at most `Q` queries that outputs a fresh
statement and a NIFS proof.

Outputs:
- `oracleProbe`: the key's PiCCS probe with oracle coins in place of
  `readK ∘ run` (`TranscriptCoverage.piCcsProbe_coins`); the response is the
  key's own;
- `test_error_le`: the oracle probe is accepted with an output that the
  witness opens, while the witness fails the source relation, with
  probability at most `(Q + 1) * IndependentExecution.testError`.

Invariant: each challenge's bad set reads the oracle only at the other
challenges of the same execution (`TranscriptCoverage.challengeCalls_injective`),
and equal calls give equal bad sets (`TranscriptCoverage.proverCalls_identify`).

The statement is adaptive; the witness and the running statement are fixed.
At a fork the extractor fixes both before the second run.

Does not own: the oracle model's fit to Poseidon2, the `Π_RLC` and `Π_DEC`
extraction, or the forking argument.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.RandomOracleTest

open scoped BigOperators
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.RandomOracle
open NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.TranscriptCoverage
open StrongReduction ConcreteCarrier

attribute [local instance low] Classical.propDecidable

/-! ## Answers -/

/-- An oracle answer: one four-word block, as the `Π_RLC` sampler reads. -/
abbrev Answer := Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Draw

/-- The extension challenge in an answer: its first two words. -/
def decodeK (answer : Answer) : K := ⟨answer ⟨0, by decide⟩, answer ⟨1, by decide⟩⟩

/-- An answer is two extension elements' worth of words. -/
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
        exact absurd inside (by unfold Spec.Folding.Nifs.NonInteractive.PiRlcSampler.drawWidth; omega)
  right_inv _ := rfl

/-- Every extension element, once: one for each pair of words. -/
def pairs : Finset K :=
  Finset.univ.map ⟨fun words : F × F => (⟨words.1, words.2⟩ : K), fun left right same => by
    cases left
    cases right
    cases same
    rfl⟩

theorem pairs_card : (pairs.card : ℝ) = (goldilocksModulus ^ 2 : Nat) := by
  simp [pairs, Fintype.card_prod, Fintype.card_fin, pow_two]

/-- The extension read of a uniform answer is uniform on `K`. -/
theorem expect_decodeK (value : K → ℝ) :
    𝔼 answer : Answer, value (decodeK answer) = 𝔼 element ∈ pairs, value element := by
  refine (Fintype.expect_equiv splitAnswer (fun answer => value (decodeK answer))
    (fun words : (F × F) × (F × F) => value ⟨words.1.1, words.1.2⟩) fun _ => rfl).trans ?_
  simp only [pairs, Finset.expect_eq_sum_div_card, Finset.sum_map, Finset.card_map,
    Finset.card_univ, Fintype.card_prod, Fintype.sum_prod_type, Finset.sum_const, nsmul_eq_mul,
    ← Finset.mul_sum, Function.Embedding.coeFn_mk]
  have positive : (0 : ℝ) < Fintype.card F := by exact_mod_cast Fintype.card_pos
  push_cast
  field_simp

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

/-! ## Oracle domain -/

private theorem exists_bound (degree : Nat) : ∃ bound : Nat,
    ∀ (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
      (proof : Proof degree) (challenge : Challenge),
      (challengeCalls fresh proof challenge).length ≤ bound := by
  by_cases inhabited : Nonempty
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits) × Proof degree)
  · obtain ⟨fresh₀, proof₀⟩ := inhabited
    refine ⟨∑ challenge, (challengeCalls fresh₀ proof₀ challenge).length,
      fun fresh proof challenge => ?_⟩
    rw [challengeCalls_length_eq fresh fresh₀ proof proof₀]
    exact Finset.single_le_sum (f := fun challenge => (challengeCalls fresh₀ proof₀ challenge).length)
      (fun _ _ => Nat.zero_le _) (Finset.mem_univ challenge)
  · exact ⟨0, fun fresh proof _ => absurd ⟨⟨fresh, proof⟩⟩ inhabited⟩

/-- A length that every challenge's calls fit in. No result depends on it. -/
noncomputable def pointBound (degree : Nat) : Nat :=
  Classical.choose (exists_bound (logicalWidth := logicalWidth) (publicFits := publicFits) degree)

/-- An oracle point: a call list no longer than `pointBound`. -/
abbrev Point (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth)
    (degree : Nat) :=
  {calls : List Call // calls.length ≤
    pointBound (logicalWidth := logicalWidth) (publicFits := publicFits) degree}

noncomputable instance (degree : Nat) : Fintype (Point logicalWidth publicFits degree) :=
  (List.finite_length_le Call _).fintype

/-- The oracle point of one challenge of one execution. -/
noncomputable def point {degree : Nat}
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof degree) (challenge : Challenge) : Point logicalWidth publicFits degree :=
  ⟨challengeCalls fresh proof challenge,
    Classical.choose_spec (exists_bound degree) fresh proof challenge⟩

/-- The extension read of a call list: the oracle answer at its point. -/
noncomputable def read {degree : Nat} (oracle : Point logicalWidth publicFits degree → Answer)
    (calls : List Call) : K :=
  if fits : calls.length ≤ pointBound (logicalWidth := logicalWidth)
      (publicFits := publicFits) degree then decodeK (oracle ⟨calls, fits⟩) else K.zero

/-- The PiCCS coins of one execution under `oracle`. -/
noncomputable def coins {degree : Nat} (oracle : Point logicalWidth publicFits degree → Answer)
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof degree) : PublicCoins K productionShape :=
  coinsFrom (read oracle) fresh proof

theorem read_challengeCalls {degree : Nat} (oracle : Point logicalWidth publicFits degree → Answer)
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof degree) (challenge : Challenge) :
    read oracle (challengeCalls fresh proof challenge) = decodeK (oracle (point fresh proof challenge)) :=
  dif_pos _

variable (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
  (witness : OutputWitness productionShape (Phi81CarrierLayout.carrierWidth logicalWidth))

local notation "Degree" => ProductionKey.degreeBound relation
local notation "Oracle" => Point logicalWidth publicFits (ProductionKey.degreeBound relation) → Answer

/-- The key's PiCCS probe with oracle coins. The key's own coins read the
sponge state instead (`TranscriptCoverage.piCcsProbe_coins`). -/
noncomputable def oracleProbe (oracle : Oracle)
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof (ProductionKey.degreeBound relation)) : Probe K productionShape :=
  { (ProductionKey.key relation ajtai).piCcsProbe running fresh proof with
    coins := coins oracle fresh proof }

/-- The oracle probe is accepted and `witness` opens its output, but `witness`
fails the source relation of the statement. -/
def FalseAcceptance (oracle : Oracle)
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof (ProductionKey.degreeBound relation)) : Prop :=
  (oracleProbe relation ajtai running oracle fresh proof).FixedWidthAccepted extensionOps K.embed
      ((ProductionKey.key relation ajtai).statement running fresh) (ProductionKey.degreeBound relation) ∧
    AmbientOutputHolds extensionOps K.embed (PaperAlgebra.openingMaps ajtai) productionGlobalParams
      ((ProductionKey.key relation ajtai).statement running fresh)
      (oracleProbe relation ajtai running oracle fresh proof) witness ∧
    ¬ SourceHolds extensionOps K.embed (PaperAlgebra.openingMaps ajtai) productionGlobalParams
      ((ProductionKey.key relation ajtai).statement running fresh) witness

/-! ## Bad sets -/

/-- The source data of `fresh` for the fixed running statement and witness. -/
noncomputable def source (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    ProtocolPolynomial.Data K productionShape :=
  ((ProductionKey.key relation ajtai).statement running fresh).sourceProtocolData K.embed witness

/-- The bad set of one coin: `RoundByRound`'s set for the source data of
`fresh`. `Π_RLC` coins have none here. -/
noncomputable def coinBad (challenge : Challenge) (coins : PublicCoins K productionShape)
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof (ProductionKey.degreeBound relation)) : Set K :=
  match challenge with
  | .alpha coordinate => RoundByRound.alphaBad
      ((source relation ajtai running witness fresh).toJointData extensionOps) coins.alpha coordinate
  | .gamma => RoundByRound.gammaBad
      ((source relation ajtai running witness fresh).toJointData extensionOps) coins.alpha
  | .round index => RoundByRound.roundBad (source relation ajtai running witness fresh)
      coins.alpha coins.gamma (coins.roundPoint.coordinates.take index.val)
      (proof.piCcsRounds index) (productionShape.cubeVariables - index.val - 1)
  | .rho _ => ∅

/-- The answers at `target` that put its challenge's coin into that coin's bad
set, for any execution whose challenge reads `target`. -/
def bad (challenge : Challenge) (target : Point logicalWidth publicFits Degree) (oracle : Oracle) :
    Set Answer :=
  {answer | ∃ fresh proof, point fresh proof challenge = target ∧
    decodeK answer ∈ coinBad relation ajtai running witness challenge (coins oracle fresh proof)
      fresh proof}

/-- The error of one challenge's bad set. -/
noncomputable def error : Challenge → ℝ
  | .alpha _ => 1 / (goldilocksModulus ^ 2 : Nat)
  | .gamma => (productionShape.jointCoefficientCount - 1 : Nat) / (goldilocksModulus ^ 2 : Nat)
  | .round _ => 9 / (goldilocksModulus ^ 2 : Nat)
  | .rho _ => 0

theorem error_nonnegative (challenge : Challenge) : 0 ≤ error challenge := by
  cases challenge <;> simp only [error] <;> positivity

/-! ## Coins after one changed answer -/

private theorem cubePoint_eq {size : Nat} {left right : CubePoint K size}
    (same : left.coordinates = right.coordinates) : left = right := by
  cases left
  cases right
  cases same
  rfl

omit relation ajtai running witness in
private theorem read_update {degree : Nat} (oracle : Point logicalWidth publicFits degree → Answer)
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof degree) {challenge other : Challenge} (different : other ≠ challenge)
    (answer : Answer) :
    read (Function.update oracle (point fresh proof challenge) answer)
        (challengeCalls fresh proof other) = read oracle (challengeCalls fresh proof other) := by
  have apart : point fresh proof other ≠ point fresh proof challenge := fun same =>
    different (challengeCalls_injective fresh proof (congrArg Subtype.val same))
  rw [read_challengeCalls, read_challengeCalls, Function.update_of_ne apart]

omit relation ajtai running witness in
/-- Changing the answer at a non-`α` point leaves `α` unchanged. -/
private theorem alpha_update {degree : Nat} (oracle : Point logicalWidth publicFits degree → Answer)
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof degree) {challenge : Challenge} (notAlpha : ∀ index, challenge ≠ .alpha index)
    (answer : Answer) :
    (coins (Function.update oracle (point fresh proof challenge) answer) fresh proof).alpha =
      (coins oracle fresh proof).alpha := by
  apply cubePoint_eq
  simp only [coins, coinsFrom]
  congr 1
  funext index
  exact read_update oracle fresh proof (fun same => notAlpha index same.symm) answer

omit relation ajtai running witness in
/-- Changing the answer at `α` coordinate `index` changes only that coordinate. -/
private theorem alpha_update_self {degree : Nat}
    (oracle : Point logicalWidth publicFits degree → Answer)
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof degree) (index : Fin productionShape.cubeVariables) (answer : Answer) :
    (coins (Function.update oracle (point fresh proof (.alpha index)) answer) fresh proof).alpha =
      ⟨(coins oracle fresh proof).alpha.coordinates.set index.val (decodeK answer),
        by simp [(coins oracle fresh proof).alpha.dimension]⟩ := by
  apply cubePoint_eq
  simp only [coins, coinsFrom]
  apply List.ext_getElem (by simp)
  intro position inside _
  simp only [List.getElem_ofFn]
  by_cases same : position = index.val
  · subst same
    simp [read_challengeCalls]
  · rw [List.getElem_set_ne (Ne.symm same), List.getElem_ofFn]
    exact read_update oracle fresh proof
      (fun equal => same (congrArg Fin.val (Challenge.alpha.inj equal))) answer

omit relation ajtai running witness in
private theorem gamma_update {degree : Nat} (oracle : Point logicalWidth publicFits degree → Answer)
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof degree) {challenge : Challenge} (notGamma : challenge ≠ .gamma)
    (answer : Answer) :
    (coins (Function.update oracle (point fresh proof challenge) answer) fresh proof).gamma =
      (coins oracle fresh proof).gamma :=
  read_update oracle fresh proof (Ne.symm notGamma) answer

omit relation ajtai running witness in
/-- Changing the answer at round `index` leaves the earlier rounds unchanged. -/
private theorem rounds_update {degree : Nat} (oracle : Point logicalWidth publicFits degree → Answer)
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof degree) (index : Fin productionShape.cubeVariables) (answer : Answer) :
    (coins (Function.update oracle (point fresh proof (.round index)) answer)
        fresh proof).roundPoint.coordinates.take index.val =
      (coins oracle fresh proof).roundPoint.coordinates.take index.val := by
  simp only [coins, coinsFrom]
  apply List.ext_getElem (by simp)
  intro position inside _
  simp only [List.getElem_take, List.getElem_ofFn]
  have below : position < index.val := by simp at inside; omega
  exact read_update oracle fresh proof (fun equal => by
    have := congrArg Fin.val (Challenge.round.inj equal)
    simp at this
    omega) answer

/-- A coin's bad set does not read that coin. -/
private theorem coinBad_update (challenge : Challenge) (oracle : Oracle)
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof (ProductionKey.degreeBound relation)) (answer : Answer) :
    coinBad relation ajtai running witness challenge
        (coins (Function.update oracle (point fresh proof challenge) answer) fresh proof)
        fresh proof =
      coinBad relation ajtai running witness challenge (coins oracle fresh proof) fresh proof := by
  cases challenge with
  | alpha index =>
      simp only [coinBad]
      rw [alpha_update_self, RoundByRound.alphaBad_set]
  | gamma =>
      simp only [coinBad]
      rw [alpha_update oracle fresh proof (fun _ => by simp)]
  | round index =>
      simp only [coinBad]
      rw [alpha_update oracle fresh proof (fun _ => by simp),
        gamma_update oracle fresh proof (by simp), rounds_update]
  | rho => rfl

/-- Each bad set reads the oracle only away from its own point. -/
theorem bad_local (challenge : Challenge) : Local (bad relation ajtai running witness challenge) := by
  intro target oracle answer
  ext candidate
  constructor
  · rintro ⟨fresh, proof, located, inside⟩
    subst located
    exact ⟨fresh, proof, rfl, by rwa [coinBad_update] at inside⟩
  · rintro ⟨fresh, proof, located, inside⟩
    subst located
    exact ⟨fresh, proof, rfl, by rwa [coinBad_update]⟩

/-! ## Mass -/

omit relation ajtai running witness in
private theorem below_of_mem_take {count bound : Nat} {index : Fin count}
    (member : index ∈ (canonicalFinIndices count).take bound) : index.val < bound := by
  obtain ⟨position, inside, located⟩ := List.getElem_of_mem member
  subst located
  simp only [List.length_take, canonicalFinIndices, List.length_ofFn] at inside
  simp only [canonicalFinIndices, List.getElem_take, List.getElem_ofFn, id]
  omega

omit relation ajtai running witness in
/-- A round's calls read the statement and the messages up to that round only. -/
private theorem challengeCalls_round_congr {degree : Nat}
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof proof' : Proof degree) (index : Fin productionShape.cubeVariables)
    (agree : ∀ earlier : Fin productionShape.cubeVariables, earlier.val ≤ index.val →
      proof.piCcsRounds earlier = proof'.piCcsRounds earlier) :
    challengeCalls fresh proof (.round index) = challengeCalls fresh proof' (.round index) := by
  have message (earlier : Fin productionShape.cubeVariables) (atMost : earlier.val ≤ index.val) :
      messages proof earlier = messages proof' earlier := by
    simp only [messages, agree earlier atMost]
  have earlierRounds :
      ((canonicalFinIndices productionShape.cubeVariables).take index.val).flatMap
          (roundCalls (messages proof)) =
        ((canonicalFinIndices productionShape.cubeVariables).take index.val).flatMap
          (roundCalls (messages proof')) := by
    simp only [List.flatMap]
    congr 1
    apply List.map_congr_left
    intro earlier member
    simp only [roundCalls, messageCalls, message earlier (below_of_mem_take member).le]
  simp only [challengeCalls, proverCalls, roundPrefixCalls, earlierRounds, messageCalls,
    message index le_rfl]

omit relation ajtai running witness in
private theorem coordinate_alpha {degree : Nat}
    (oracle : Point logicalWidth publicFits degree → Answer)
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof degree) (index : Fin productionShape.cubeVariables) :
    RoundByRound.coordinate (coins oracle fresh proof).alpha index =
      decodeK (oracle (point fresh proof (.alpha index))) := by
  simp only [RoundByRound.coordinate, coins, coinsFrom, List.getElem_ofFn, read_challengeCalls]

omit relation ajtai running witness in
private theorem coordinate_round {degree : Nat}
    (oracle : Point logicalWidth publicFits degree → Answer)
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof degree) (index : Fin productionShape.cubeVariables) :
    RoundByRound.coordinate (coins oracle fresh proof).roundPoint index =
      decodeK (oracle (point fresh proof (.round index))) := by
  simp only [RoundByRound.coordinate, coins, coinsFrom, List.getElem_ofFn, read_challengeCalls]

omit relation ajtai running witness in
/-- `α` and `γ` read only the statement calls, not the proof. -/
private theorem alpha_gamma_proof_irrelevant {degree : Nat}
    (oracle : Point logicalWidth publicFits degree → Answer)
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof proof' : Proof degree) :
    (coins oracle fresh proof).alpha = (coins oracle fresh proof').alpha ∧
      (coins oracle fresh proof).gamma = (coins oracle fresh proof').gamma := by
  have alpha (index : Fin productionShape.cubeVariables) :
      challengeCalls fresh proof (.alpha index) = challengeCalls fresh proof' (.alpha index) := rfl
  have gamma : challengeCalls fresh proof .gamma = challengeCalls fresh proof' .gamma := rfl
  refine ⟨cubePoint_eq ?_, ?_⟩
  · simp only [coins, coinsFrom, alpha]
  · simp only [coins, coinsFrom, gamma]

/-- Two executions whose challenge reads the same point have the same bad set
there. -/
private theorem coinBad_congr (challenge : Challenge) (oracle : Oracle)
    {fresh fresh' : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)}
    {proof proof' : Proof (ProductionKey.degreeBound relation)}
    (same : point fresh proof challenge = point fresh' proof' challenge) :
    coinBad relation ajtai running witness challenge (coins oracle fresh proof) fresh proof =
      coinBad relation ajtai running witness challenge (coins oracle fresh' proof') fresh' proof' := by
  have calls : challengeCalls fresh proof challenge = challengeCalls fresh' proof' challenge :=
    congrArg Subtype.val same
  have agree := proverCalls_identify challenge (List.append_cancel_right calls)
  cases challenge with
  | alpha index =>
      have equal : fresh = fresh' := agree
      subst equal
      simp only [coinBad]
      rw [(alpha_gamma_proof_irrelevant oracle fresh proof proof').1]
  | gamma =>
      have equal : fresh = fresh' := agree
      subst equal
      simp only [coinBad]
      rw [(alpha_gamma_proof_irrelevant oracle fresh proof proof').1]
  | round index =>
      obtain ⟨equal, rounds⟩ := agree
      subst equal
      have earlier : (coins oracle fresh proof).roundPoint.coordinates.take index.val =
          (coins oracle fresh proof').roundPoint.coordinates.take index.val := by
        simp only [coins, coinsFrom]
        apply List.ext_getElem (by simp)
        intro position inside _
        have below : position < index.val := by simp at inside; omega
        simp only [List.getElem_take, List.getElem_ofFn]
        rw [challengeCalls_round_congr fresh proof proof' ⟨position, by omega⟩
          fun round atMost => rounds round (by simp at atMost; omega)]
      simp only [coinBad]
      rw [earlier, rounds index le_rfl, (alpha_gamma_proof_irrelevant oracle fresh proof proof').1,
        (alpha_gamma_proof_irrelevant oracle fresh proof proof').2]
  | rho => simp only [coinBad]

private theorem coinBad_probability_le (challenge : Challenge) (coins : PublicCoins K productionShape)
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof (ProductionKey.degreeBound relation)) :
    (𝔼 element ∈ pairs,
      if element ∈ coinBad relation ajtai running witness challenge coins fresh proof then (1 : ℝ)
        else 0) ≤ error challenge := by
  cases challenge with
  | alpha index =>
      calc
        _ = 𝔼 element ∈ pairs, if element ∈ RoundByRound.alphaBad
              ((source relation ajtai running witness fresh).toJointData extensionOps)
              coins.alpha index then (1 : ℝ) else 0 :=
          Finset.expect_congr rfl fun _ _ => if_congr Iff.rfl rfl rfl
        _ ≤ 1 / (pairs.card : ℝ) := RoundByRound.alphaBad_probability_le pairs _ coins.alpha index
        _ = error (.alpha index) := by rw [pairs_card]; rfl
  | gamma =>
      calc
        _ = 𝔼 element ∈ pairs, if element ∈ RoundByRound.gammaBad
              ((source relation ajtai running witness fresh).toJointData extensionOps)
              coins.alpha then (1 : ℝ) else 0 :=
          Finset.expect_congr rfl fun _ _ => if_congr Iff.rfl rfl rfl
        _ ≤ (productionShape.jointCoefficientCount - 1 : Nat) / (pairs.card : ℝ) :=
          RoundByRound.gammaBad_probability_le pairs _ coins.alpha
        _ = error .gamma := by rw [pairs_card]; rfl
  | round index =>
      have degreeCovers :
          (source relation ajtai running witness fresh).toVerifierInput.sumcheckDegreeBound ≤
            ProductionKey.degreeBound relation :=
        (ProductionKey.key relation ajtai).statement_sumcheckDegreeBound_le running fresh
      have length : (coins.roundPoint.coordinates.take index.val).length + 1 +
          (productionShape.cubeVariables - index.val - 1) = productionShape.cubeVariables := by
        have below := index.isLt
        simp only [List.length_take, coins.roundPoint.dimension]
        omega
      have bound := RoundByRound.roundBad_probability_le pairs _ coins.alpha coins.gamma
        degreeCovers _ (proof.piCcsRounds index) _ length
      calc
        _ = 𝔼 element ∈ pairs, if element ∈ RoundByRound.roundBad
              (source relation ajtai running witness fresh) coins.alpha coins.gamma
              (coins.roundPoint.coordinates.take index.val) (proof.piCcsRounds index)
              (productionShape.cubeVariables - index.val - 1) then (1 : ℝ) else 0 :=
          Finset.expect_congr rfl fun _ _ => if_congr Iff.rfl rfl rfl
        _ ≤ _ := bound
        _ = error (.round index) := by
          rw [pairs_card, ProductionKey.degreeBound_eq]
          simp [error]
  | rho =>
      simp only [coinBad, error, Set.mem_empty_iff_false, if_false, Finset.expect_const_zero,
        le_refl]

/-- Each bad set has probability at most its challenge's error. -/
theorem bad_mass_le (challenge : Challenge) (target : Point logicalWidth publicFits Degree)
    (oracle : Oracle) :
    mass (bad relation ajtai running witness challenge target oracle) ≤ error challenge := by
  by_cases located : ∃ fresh proof, point fresh proof challenge = target
  · obtain ⟨fresh₀, proof₀, rfl⟩ := located
    have inside : bad relation ajtai running witness challenge (point fresh₀ proof₀ challenge) oracle ⊆
        {answer | decodeK answer ∈ coinBad relation ajtai running witness challenge
          (coins oracle fresh₀ proof₀) fresh₀ proof₀} := by
      rintro answer ⟨fresh, proof, same, member⟩
      rwa [coinBad_congr relation ajtai running witness challenge oracle same] at member
    refine (mass_mono inside).trans ?_
    have uniform := expect_decodeK (fun element => if element ∈ coinBad relation ajtai running
      witness challenge (coins oracle fresh₀ proof₀) fresh₀ proof₀ then (1 : ℝ) else 0)
    have bound := coinBad_probability_le relation ajtai running witness challenge
      (coins oracle fresh₀ proof₀) fresh₀ proof₀
    unfold mass
    exact (le_of_eq uniform).trans bound
  · have empty : bad relation ajtai running witness challenge target oracle = ∅ := by
      ext answer
      simp only [bad, Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
      rintro ⟨fresh, proof, same, _⟩
      exact located ⟨fresh, proof, same⟩
    rw [empty]
    simp only [mass, Set.mem_empty_iff_false, if_false, Finset.expect_const_zero]
    exact error_nonnegative challenge

/-! ## Test error -/

omit relation ajtai running witness in
private theorem indicator_le_sum {Index : Type} [Fintype Index] (event : Index → Prop)
    (chosen : Index) (holds : event chosen) :
    (1 : ℝ) ≤ ∑ index, if event index then (1 : ℝ) else 0 := by
  have single := Finset.single_le_sum (f := fun index => if event index then (1 : ℝ) else 0)
    (fun index _ => by split <;> norm_num) (Finset.mem_univ chosen)
  simpa only [if_pos holds] using single

omit relation ajtai running witness in
private theorem indicator_sum_nonnegative {Index : Type} [Fintype Index] (event : Index → Prop) :
    (0 : ℝ) ≤ ∑ index, if event index then (1 : ℝ) else 0 :=
  Finset.sum_nonneg fun index _ => by split <;> norm_num

/-- The `α`, `γ` and round coins of a false acceptance: one of them hits its
bad set at its own point. -/
private theorem falseAcceptance_hits (oracle : Oracle)
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof (ProductionKey.degreeBound relation))
    (false_ : FalseAcceptance relation ajtai running witness oracle fresh proof) :
    (∃ index, oracle (point fresh proof (.alpha index)) ∈
        bad relation ajtai running witness (.alpha index) (point fresh proof (.alpha index)) oracle) ∨
      oracle (point fresh proof .gamma) ∈
        bad relation ajtai running witness .gamma (point fresh proof .gamma) oracle ∨
      ∃ index, oracle (point fresh proof (.round index)) ∈
        bad relation ajtai running witness (.round index) (point fresh proof (.round index))
          oracle := by
  obtain ⟨accepted, ambient, invalid⟩ := false_
  rcases RoundByRound.falseAcceptance_splits (PaperAlgebra.openingMaps ajtai) productionGlobalParams
      rfl ((ProductionKey.key relation ajtai).statement running fresh)
      (ProductionKey.key relation ajtai).constantLaw
      ((ProductionKey.key relation ajtai).statement_sumcheckDegreeBound_le running fresh) witness
      invalid _ ambient accepted with ⟨index, member⟩ | member | ⟨certificate, index, message,
        decoded, located, member⟩
  · refine Or.inl ⟨index, fresh, proof, rfl, ?_⟩
    change RoundByRound.coordinate (coins oracle fresh proof).alpha index ∈ _ at member
    rwa [coordinate_alpha] at member
  · refine Or.inr (Or.inl ⟨fresh, proof, rfl, ?_⟩)
    change read oracle (challengeCalls fresh proof .gamma) ∈ _ at member
    rwa [read_challengeCalls] at member
  · refine Or.inr (Or.inr ⟨index, fresh, proof, rfl, ?_⟩)
    have response : (oracleProbe relation ajtai running oracle fresh proof).response.rounds =
        SumCheck.Finite.FixedPhase.RawCertificate.encode
          ((ProductionKey.key relation ajtai).piCcsFixedCertificate running fresh proof) :=
      Spec.Folding.Nifs.PaperNonInteractive.piCcsProbe_rounds _ running fresh proof
    rw [response, SumCheck.Finite.FixedPhase.RawCertificate.decode_encode] at decoded
    cases decoded
    have equal : message = proof.piCcsRounds index := by
      simpa [Spec.Folding.Nifs.PaperNonInteractive.Key.piCcsFixedCertificate] using located.symm
    subst equal
    change RoundByRound.coordinate (coins oracle fresh proof).roundPoint index ∈ _ at member
    rwa [coordinate_round] at member

/-- PiCCS test error in the random-oracle model. An adaptive adversary with at
most `queries` oracle queries outputs a fresh statement and a proof; the
oracle probe of that output is a false acceptance for the fixed witness with
probability at most `(queries + 1) * testError`. -/
theorem test_error_le {Output : Type}
    (adversary : OracleComp (Point logicalWidth publicFits (ProductionKey.degreeBound relation))
      Answer Output)
    {queries : Nat} (bounded : adversary.QueryBound queries)
    (fresh : Output → Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Output → Proof (ProductionKey.degreeBound relation)) :
    𝔼 oracle, (if FalseAcceptance relation ajtai running witness oracle
        (fresh (adversary.run oracle)) (proof (adversary.run oracle)) then (1 : ℝ) else 0) ≤
      (queries + 1) * IndependentExecution.testError productionShape 9 := by
  let hit (challenge : Challenge) (oracle : Oracle) : Prop :=
    oracle (point (fresh (adversary.run oracle)) (proof (adversary.run oracle)) challenge) ∈
      bad relation ajtai running witness challenge
        (point (fresh (adversary.run oracle)) (proof (adversary.run oracle)) challenge) oracle
  have pinned (challenge : Challenge) :
      𝔼 oracle, (if hit challenge oracle then (1 : ℝ) else 0) ≤ (queries + 1) * error challenge :=
    pinned_le (bad relation ajtai running witness challenge)
      (bad_local relation ajtai running witness challenge) (error challenge)
      (bad_mass_le relation ajtai running witness challenge) (error_nonnegative challenge) bounded
      (fun output => point (fresh output) (proof output) challenge)
  have split (oracle : Oracle) :
      (if FalseAcceptance relation ajtai running witness oracle
          (fresh (adversary.run oracle)) (proof (adversary.run oracle)) then (1 : ℝ) else 0) ≤
        (∑ index, if hit (.alpha index) oracle then (1 : ℝ) else 0) +
          (if hit .gamma oracle then (1 : ℝ) else 0) +
          ∑ index, if hit (.round index) oracle then (1 : ℝ) else 0 := by
    have alphas := indicator_sum_nonnegative fun index => hit (.alpha index) oracle
    have rounds := indicator_sum_nonnegative fun index => hit (.round index) oracle
    have gamma : (0 : ℝ) ≤ if hit .gamma oracle then (1 : ℝ) else 0 := by split <;> norm_num
    by_cases false_ : FalseAcceptance relation ajtai running witness oracle
        (fresh (adversary.run oracle)) (proof (adversary.run oracle))
    · rw [if_pos false_]
      rcases falseAcceptance_hits relation ajtai running witness oracle _ _ false_ with
        ⟨index, holds⟩ | holds | ⟨index, holds⟩
      · have := indicator_le_sum (fun index => hit (.alpha index) oracle) index holds
        linarith
      · have : (1 : ℝ) ≤ if hit .gamma oracle then (1 : ℝ) else 0 := by
          rw [if_pos holds]
        linarith
      · have := indicator_le_sum (fun index => hit (.round index) oracle) index holds
        linarith
    · rw [if_neg false_]
      linarith
  have alphaBound : 𝔼 oracle, (∑ index, if hit (.alpha index) oracle then (1 : ℝ) else 0) ≤
      productionShape.cubeVariables * ((queries + 1) * error (.alpha ⟨0, by decide⟩)) := by
    rw [Finset.expect_sum_comm]
    refine (Finset.sum_le_sum fun index _ => pinned (.alpha index)).trans (le_of_eq ?_)
    simp [error]
  have roundBound : 𝔼 oracle, (∑ index, if hit (.round index) oracle then (1 : ℝ) else 0) ≤
      productionShape.cubeVariables * ((queries + 1) * error (.round ⟨0, by decide⟩)) := by
    rw [Finset.expect_sum_comm]
    refine (Finset.sum_le_sum fun index _ => pinned (.round index)).trans (le_of_eq ?_)
    simp [error]
  have total :
      𝔼 oracle, (if FalseAcceptance relation ajtai running witness oracle
          (fresh (adversary.run oracle)) (proof (adversary.run oracle)) then (1 : ℝ) else 0) ≤
        (𝔼 oracle, ∑ index, if hit (.alpha index) oracle then (1 : ℝ) else 0) +
          (𝔼 oracle, if hit .gamma oracle then (1 : ℝ) else 0) +
          𝔼 oracle, ∑ index, if hit (.round index) oracle then (1 : ℝ) else 0 := by
    refine (Finset.expect_le_expect fun oracle _ => split oracle).trans (le_of_eq ?_)
    rw [Finset.expect_add_distrib, Finset.expect_add_distrib]
  refine total.trans ((add_le_add (add_le_add alphaBound (pinned .gamma)) roundBound).trans
    (le_of_eq ?_))
  have cube : (productionShape.cubeVariables : ℝ) = 28 := by
    exact_mod_cast (rfl : productionShape.cubeVariables = 28)
  have coefficients : productionShape.jointCoefficientCount - 1 + productionShape.cubeVariables =
      (productionShape.jointCoefficientCount - 1) + 28 := rfl
  simp only [error, IndependentExecution.testError, coefficients, cube]
  push_cast
  ring

end NightstreamFPrime.Lifecycle.RandomOracleTest

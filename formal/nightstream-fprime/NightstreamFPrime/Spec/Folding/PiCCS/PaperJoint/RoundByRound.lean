import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.GoldilocksCausal
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SignedMixingProbability

/-!
Owns the round-by-round split of a false PiCCS acceptance: one bad set for
each verifier coin (every `α` coordinate, `γ`, and every SumCheck round).

Inputs: a statement, a fixed witness that fails the source relation, and an
accepted probe whose output that witness opens.

Outputs:
- `falseAcceptance_splits`: some coin lies in its bad set;
- `alphaBad_probability_le`, `gammaBad_probability_le`,
  `roundBad_probability_le`: a coin drawn uniformly from `samples` lies in its
  bad set with probability at most `1`, `J - 1` or `width` over the sample
  count. Over all of `K` and summed over the coins, these are
  `IndependentExecution.testError`.

Invariant: a bad set does not read its own coin. The `α` set replaces its
coordinate (`alphaBad_set`); the `γ` set reads only `α`; a round set reads
only `α`, `γ` and the earlier rounds.

The `α` split follows the table recursion of `BooleanMixingProbability` along
one fixed path of nonzero sub-tables.

Does not own: the coin law. `Lifecycle/RandomOracleTest.lean` draws the
coins from a random oracle.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.RoundByRound

open scoped BigOperators
open NightstreamFPrime.Spec SumCheck.Finite
open ConcreteCarrier StrongReduction SignedCoefficientObject
open _root_.NightstreamFPrime.Spec.SumCheck.Finite.GoldilocksCausal (Hit hit_probability_le)

attribute [local instance] Classical.propDecidable

/-- One coordinate of a cube point. -/
def coordinate {size : Nat} (point : CubePoint K size) (index : Fin size) : K :=
  point.coordinates[index.val]'(by rw [point.dimension]; exact index.isLt)

theorem set_coordinate {size : Nat} (point : CubePoint K size) (index : Fin size) :
    point.coordinates.set index.val (coordinate point index) = point.coordinates :=
  List.set_getElem_self _

/-! ## `α`: one fixed path of nonzero sub-tables -/

/-- The low half, unless it is all zero. -/
noncomputable def selectedChild {size : Nat} (low high : BooleanTable K size) :
    BooleanTable K size :=
  if low.AllEntriesZero extensionOps then high else low

/-- Along the selected path, the halves of the sub-table at `depth` are not both
zero at the later coordinates, and coordinate `depth` makes the sub-table zero. -/
noncomputable def VanishesAt : {size : Nat} → BooleanTable K size → List K → Nat → Prop
  | _ + 1, .branch low high, value :: rest, 0 =>
      ¬ (low.evaluateCoordinates extensionOps rest = K.zero ∧
          high.evaluateCoordinates extensionOps rest = K.zero) ∧
        extensionOps.add (low.evaluateCoordinates extensionOps rest)
          (extensionOps.mul value (extensionOps.sub
            (high.evaluateCoordinates extensionOps rest)
            (low.evaluateCoordinates extensionOps rest))) = K.zero
  | _ + 1, .branch low high, _ :: rest, depth + 1 =>
      VanishesAt (selectedChild low high) rest depth
  | _, _, _, _ => False

/-- A nonzero table that is zero at a point vanishes first at some coordinate
of the selected path. -/
theorem exists_vanishesAt : {size : Nat} → (table : BooleanTable K size) →
    (coordinates : List K) → coordinates.length = size →
    ¬ table.AllEntriesZero extensionOps →
    table.evaluateCoordinates extensionOps coordinates = K.zero →
    ∃ depth < size, VanishesAt table coordinates depth
  | 0, .leaf value, [], _, nonzero, zero => by
      exact absurd (fun entry member => by
        simp only [BooleanTable.entries, List.mem_singleton] at member
        subst member
        exact zero) nonzero
  | size + 1, .branch low high, value :: rest, length, nonzero, zero => by
      by_cases both : low.evaluateCoordinates extensionOps rest = K.zero ∧
          high.evaluateCoordinates extensionOps rest = K.zero
      · have childNonzero : ¬ (selectedChild low high).AllEntriesZero extensionOps := by
          unfold selectedChild
          split
          · rename_i lowZero
            intro highZero
            apply nonzero
            intro entry member
            rcases List.mem_append.mp member with inside | inside
            · exact lowZero entry inside
            · exact highZero entry inside
          · assumption
        have childZero :
            (selectedChild low high).evaluateCoordinates extensionOps rest = K.zero := by
          unfold selectedChild
          split
          · exact both.2
          · exact both.1
        obtain ⟨depth, below, vanishes⟩ := exists_vanishesAt (selectedChild low high) rest
          (by simpa using length) childNonzero childZero
        exact ⟨depth + 1, by omega, vanishes⟩
      · exact ⟨0, by omega, both, zero⟩
  | 0, .leaf _, _ :: _, length, _, _ => by simp at length
  | _ + 1, .branch _ _, [], length, _, _ => by simp at length

/-- For fixed other coordinates, at most one value of coordinate `depth` makes
the selected path vanish there. -/
theorem vanishesAt_probability_le (samples : Finset K) : {size : Nat} →
    (table : BooleanTable K size) → (coordinates : List K) → (depth : Nat) →
    (𝔼 value ∈ samples,
      if VanishesAt table (coordinates.set depth value) depth then (1 : ℝ) else 0) ≤
      1 / (samples.card : ℝ)
  | _ + 1, .branch low high, _ :: rest, 0 => by
      simp only [List.set_cons_zero, VanishesAt]
      by_cases both : low.evaluateCoordinates extensionOps rest = K.zero ∧
          high.evaluateCoordinates extensionOps rest = K.zero
      · simp only [both.1, both.2, and_self, not_true_eq_false, false_and, if_false,
          Finset.expect_const_zero]
        positivity
      · simp only [both, not_false_eq_true, true_and]
        have coefficient : ∃ entry ∈ [low.evaluateCoordinates extensionOps rest,
            extensionOps.sub (high.evaluateCoordinates extensionOps rest)
              (low.evaluateCoordinates extensionOps rest)], entry ≠ K.zero := by
          by_cases lowZero : low.evaluateCoordinates extensionOps rest = K.zero
          · have highNonzero : high.evaluateCoordinates extensionOps rest ≠ K.zero :=
              fun highZero => both ⟨lowZero, highZero⟩
            have subZero : extensionOps.sub (high.evaluateCoordinates extensionOps rest)
                K.zero = high.evaluateCoordinates extensionOps rest := by
              change extensionOps.add _ (extensionOps.neg extensionOps.zero) = _
              rw [extensionZeroLaws.neg_zero, extensionLaws.add_zero]
            exact ⟨extensionOps.sub (high.evaluateCoordinates extensionOps rest)
              (low.evaluateCoordinates extensionOps rest), by simp,
              by simpa [lowZero, subZero] using highNonzero⟩
          · exact ⟨low.evaluateCoordinates extensionOps rest, by simp, lowZero⟩
        have evaluation (point : K) :
            Message.evaluateCoefficients GoldilocksRoots.ops point
                [low.evaluateCoordinates extensionOps rest,
                  extensionOps.sub (high.evaluateCoordinates extensionOps rest)
                    (low.evaluateCoordinates extensionOps rest)] =
              extensionOps.add (low.evaluateCoordinates extensionOps rest)
                (extensionOps.mul point (extensionOps.sub
                  (high.evaluateCoordinates extensionOps rest)
                  (low.evaluateCoordinates extensionOps rest))) := by
          change extensionOps.add _ (extensionOps.mul point (extensionOps.add _
            (extensionOps.mul point extensionOps.zero))) = _
          rw [extensionLaws.mul_zero, extensionLaws.add_zero]
        simpa only [evaluation, List.length_cons, List.length_nil, Nat.reduceAdd,
          Nat.reduceSub, Nat.cast_one] using
          SignedMixingRoots.coefficient_root_probability_le _ samples coefficient
  | _ + 1, .branch low high, _ :: rest, depth + 1 => by
      simp only [List.set_cons_succ, VanishesAt]
      exact vanishesAt_probability_le samples (selectedChild low high) rest depth
  | 0, .leaf _, coordinates, depth => by
      simp only [VanishesAt, if_false, Finset.expect_const_zero]
      positivity
  | _ + 1, .branch _ _, [], depth => by
      simp only [List.set_nil, VanishesAt, if_false, Finset.expect_const_zero]
      positivity

/-- The first coefficient that is not identically zero. -/
noncomputable def selected {shape : Shape} (data : SignedJointIdentity.JointData K shape) :
    Option (Coefficient K shape) :=
  (coefficients extensionOps data).find? fun coefficient => decide ¬ coefficient.Zero extensionOps

/-- The Boolean table of the selected coefficient, when that coefficient
depends on `α`. -/
noncomputable def selectedTable {shape : Shape} (data : SignedJointIdentity.JointData K shape) :
    Option (BooleanTable K shape.cubeVariables) :=
  match selected data with
  | some (.negativeAlpha polynomial) =>
      if found : ∃ table : BooleanTable K shape.cubeVariables,
          polynomial = table.toAlphaPolynomial extensionOps then some found.choose else none
  | _ => none

/-- Coordinate `index` of `α` first makes the selected table vanish. The set
replaces that coordinate, so it does not read it. -/
def alphaBad {shape : Shape} (data : SignedJointIdentity.JointData K shape)
    (alpha : CubePoint K shape.cubeVariables) (index : Fin shape.cubeVariables) : Set K :=
  {value | ∃ table, selectedTable data = some table ∧
    VanishesAt table (alpha.coordinates.set index.val value) index.val}

/-- `γ` is a root of the nonzero specialized coefficient list. -/
def gammaBad {shape : Shape} (data : SignedJointIdentity.JointData K shape)
    (alpha : CubePoint K shape.cubeVariables) : Set K :=
  {gamma | (∃ coefficient ∈ SignedCoefficientPolynomial.coefficients extensionOps data alpha,
      coefficient ≠ K.zero) ∧
    (SignedCoefficientPolynomial.polynomial extensionOps data alpha).evaluate
      extensionOps.toOps gamma = K.zero}

/-- A wrong round message agrees with the true round polynomial at the round
challenge. -/
def roundBad {shape : Shape} {width : Nat} (data : ProtocolPolynomial.Data K shape)
    (alpha : CubePoint K shape.cubeVariables) (gamma : K) (before : List K)
    (message : FixedPolynomial K width) (remaining : Nat) : Set K :=
  {challenge | Hit (ProtocolPolynomial.polynomial extensionOps data alpha gamma) before remaining
    message challenge}

theorem alphaBad_set {shape : Shape} (data : SignedJointIdentity.JointData K shape)
    (alpha : CubePoint K shape.cubeVariables) (index : Fin shape.cubeVariables) (value : K) :
    alphaBad data ⟨alpha.coordinates.set index.val value, by simp [alpha.dimension]⟩ index =
      alphaBad data alpha index := by
  simp only [alphaBad, List.set_set]

/-! ## Split -/

/-- A signed mixing root puts some `α` coordinate or `γ` in its bad set. -/
theorem mixingRoot_splits {shape : Shape} (data : SignedJointIdentity.JointData K shape)
    (alpha : CubePoint K shape.cubeVariables) (gamma : K)
    (root : MixingRoot extensionOps data alpha gamma) :
    (∃ index, coordinate alpha index ∈ alphaBad data alpha index) ∨
      gamma ∈ gammaBad data alpha := by
  have exists_nonzero : ∃ coefficient ∈ coefficients extensionOps data,
      ¬ coefficient.Zero extensionOps := by
    by_contra absent
    apply root.coefficientNonzero
    intro coefficient inside
    by_contra nonzero
    exact absent ⟨coefficient, inside, nonzero⟩
  obtain ⟨first, firstInside, firstNonzero⟩ := exists_nonzero
  obtain ⟨chosen, chosenEq⟩ : ∃ chosen, selected data = some chosen :=
    Option.isSome_iff_exists.mp (List.find?_isSome.mpr ⟨first, firstInside, by simpa using firstNonzero⟩)
  have chosenInside : chosen ∈ coefficients extensionOps data := List.mem_of_find?_eq_some chosenEq
  have chosenNonzero : ¬ chosen.Zero extensionOps := by simpa using List.find?_some chosenEq
  by_cases zero : chosen.specialize extensionOps alpha = K.zero
  · left
    cases chosen with
    | scalar value => exact absurd zero chosenNonzero
    | negativeAlpha polynomial =>
        have found : ∃ table : BooleanTable K shape.cubeVariables,
            polynomial = table.toAlphaPolynomial extensionOps :=
          SignedMixingProbability.negative_coefficient_is_table data polynomial chosenInside
        let table := found.choose
        have represents : polynomial = table.toAlphaPolynomial extensionOps := found.choose_spec
        have tableEq : selectedTable data = some table := by
          simp only [selectedTable, chosenEq, dif_pos found]
          rfl
        have tableNonzero : ¬ table.AllEntriesZero extensionOps := by
          intro allZero
          apply chosenNonzero
          change polynomial.CoefficientZero extensionOps.toOps
          rw [represents]
          exact (BooleanTable.toAlphaPolynomial_coefficientZero_iff_allEntriesZero extensionOps
            extensionZeroLaws table).mpr allZero
        have tableZero : table.evaluateCoordinates extensionOps alpha.coordinates = K.zero := by
          have negated : extensionOps.neg (polynomial.evaluate extensionOps.toOps alpha) = K.zero :=
            zero
          rw [SignedMixingProbability.neg_zero_iff, represents,
            BooleanTable.toAlphaPolynomial_evaluate_eq_evaluate extensionOps extensionLaws] at negated
          exact negated
        obtain ⟨depth, below, vanishes⟩ := exists_vanishesAt table alpha.coordinates
          alpha.dimension tableNonzero tableZero
        refine ⟨⟨depth, below⟩, table, tableEq, ?_⟩
        rw [set_coordinate alpha ⟨depth, below⟩]
        exact vanishes
  · right
    refine ⟨⟨chosen.specialize extensionOps alpha, ?_, zero⟩, root.sampledZero⟩
    rw [← specializedCoefficients_eq extensionOps extensionLaws]
    exact List.mem_map.mpr ⟨chosen, chosenInside, rfl⟩

universe uCommitment uPublicInput

/-- A false acceptance puts some verifier coin in its bad set: an `α`
coordinate, `γ`, or the challenge of a round whose message is wrong. -/
theorem falseAcceptance_splits
    {Commitment : Type uCommitment} {PublicInput : Type uPublicInput}
    {shape : Shape} {columns blockCount width : Nat}
    (openingMaps : OpeningMaps Commitment PublicInput columns) (params : GlobalParams)
    (freshBound : params.b = 2)
    (statement : Statement K Commitment PublicInput shape columns blockCount baseOps)
    (constantLaw : MatrixCoefficientSource.ConstantTermLaw baseOps statement.matrixSource.kernel)
    (degreeCovers : (statement.verifierInput K.embed).sumcheckDegreeBound ≤ width)
    (witness : OutputWitness shape columns)
    (sourceInvalid : ¬ SourceHolds extensionOps K.embed openingMaps params statement witness)
    (probe : Probe K shape)
    (ambient : AmbientOutputHolds extensionOps K.embed openingMaps params statement probe witness)
    (accepted : probe.FixedWidthAccepted extensionOps K.embed statement width) :
    (∃ index, coordinate probe.coins.alpha index ∈
        alphaBad ((statement.sourceProtocolData K.embed witness).toJointData extensionOps)
          probe.coins.alpha index) ∨
      probe.coins.gamma ∈
        gammaBad ((statement.sourceProtocolData K.embed witness).toJointData extensionOps)
          probe.coins.alpha ∨
      ∃ (certificate : FixedPhase.Certificate K width) (round : Fin shape.cubeVariables)
        (message : FixedPolynomial K width),
        FixedPhase.RawCertificate.decode width probe.response.rounds = some certificate ∧
        certificate.rounds[round.val]? = some message ∧
        coordinate probe.coins.roundPoint round ∈
          roundBad (statement.sourceProtocolData K.embed witness) probe.coins.alpha
            probe.coins.gamma (probe.coins.roundPoint.coordinates.take round.val) message
            (shape.cubeVariables - round.val - 1) := by
  rcases fixedWidthAcceptedProbe_extracts_source_or_badEvent
      baseLaws baseZeroAgreement GoldilocksPrime.baseFieldNoZeroDivisors
      extensionOps extensionLaws extensionZeroLaws K.embed protocolLift openingMaps params
      freshBound statement constantLaw width degreeCovers (goldilocksModulus ^ 2)
      probe witness ambient accepted with source | mixing | failure
  · exact absurd source sourceInvalid
  · rcases mixingRoot_splits _ _ _ mixing with alpha | gamma
    · exact Or.inl alpha
    · exact Or.inr (Or.inl gamma)
  · right
    right
    obtain ⟨certificate, decoded, bad⟩ := failure
    obtain ⟨before, challenge, after, beforeMessages, message, afterMessages,
        challengesEqual, messagesEqual, length, different, equal⟩ :=
      FixedPhase.badChallenge_implies_causal_decomposition _ _ _ _ _ certificate bad
    have total : before.length + 1 + after.length = shape.cubeVariables := by
      have dimension := probe.coins.roundPoint.dimension
      rw [challengesEqual] at dimension
      simpa [Nat.add_assoc, Nat.add_comm 1] using dimension
    let round : Fin shape.cubeVariables := ⟨before.length, by omega⟩
    refine ⟨certificate, round, message, decoded, ?_, ?_⟩
    · show certificate.rounds[before.length]? = some message
      rw [messagesEqual, length]
      simp
    · have point : coordinate probe.coins.roundPoint round = challenge := by
        simp only [coordinate, round]
        simp [challengesEqual]
      have prefixEq : probe.coins.roundPoint.coordinates.take round.val = before := by
        simp [round, challengesEqual]
      have remaining : shape.cubeVariables - round.val - 1 = after.length := by
        simp only [round]
        omega
      rw [point, prefixEq, remaining]
      exact ⟨Function.ne_iff.mp different, equal⟩

/-! ## Counts -/

theorem alphaBad_probability_le (samples : Finset K) {shape : Shape}
    (data : SignedJointIdentity.JointData K shape)
    (alpha : CubePoint K shape.cubeVariables) (index : Fin shape.cubeVariables) :
    (𝔼 value ∈ samples, if value ∈ alphaBad data alpha index then (1 : ℝ) else 0) ≤
      1 / (samples.card : ℝ) := by
  cases chosen : selectedTable data with
  | none =>
      simp only [alphaBad, chosen, reduceCtorEq, false_and, exists_false, Set.setOf_false,
        Set.mem_empty_iff_false, if_false, Finset.expect_const_zero]
      positivity
  | some table =>
      simp only [alphaBad, chosen, Option.some.injEq, exists_eq_left', Set.mem_setOf_eq]
      exact vanishesAt_probability_le samples table alpha.coordinates index.val

theorem gammaBad_probability_le (samples : Finset K) {shape : Shape}
    (data : SignedJointIdentity.JointData K shape) (alpha : CubePoint K shape.cubeVariables) :
    (𝔼 gamma ∈ samples, if gamma ∈ gammaBad data alpha then (1 : ℝ) else 0) ≤
      (shape.jointCoefficientCount - 1 : Nat) / (samples.card : ℝ) := by
  by_cases nonzero : ∃ coefficient ∈ SignedCoefficientPolynomial.coefficients extensionOps data alpha,
      coefficient ≠ K.zero
  · refine le_trans (Finset.expect_le_expect fun gamma _ => ?_)
      (SignedMixingRoots.signed_gamma_probability_le data alpha samples nonzero)
    by_cases inside : gamma ∈ gammaBad data alpha
    · simp [if_pos inside, if_pos inside.2]
    · rw [if_neg inside]
      split <;> norm_num
  · simp only [gammaBad, nonzero, false_and, Set.setOf_false, Set.mem_empty_iff_false, if_false,
      Finset.expect_const_zero]
    positivity

theorem roundBad_probability_le (samples : Finset K) {shape : Shape} {width : Nat}
    (data : ProtocolPolynomial.Data K shape) (alpha : CubePoint K shape.cubeVariables) (gamma : K)
    (degreeCovers : data.toVerifierInput.sumcheckDegreeBound ≤ width) (before : List K)
    (message : FixedPolynomial K width) (remaining : Nat)
    (length : before.length + 1 + remaining = shape.cubeVariables) :
    (𝔼 challenge ∈ samples,
      if challenge ∈ roundBad data alpha gamma before message remaining then (1 : ℝ) else 0) ≤
      width / (samples.card : ℝ) := by
  obtain ⟨semantic, represents⟩ := GoldilocksCausal.sequentialRoundRepresentable data alpha gamma
    width degreeCovers before remaining length
  exact hit_probability_le samples _ before remaining message semantic represents

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.RoundByRound

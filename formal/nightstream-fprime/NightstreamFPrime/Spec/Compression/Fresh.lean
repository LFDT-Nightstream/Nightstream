import NightstreamFPrime.Spec.Compression.Game.Restoration
import NightstreamFPrime.Spec.Compression.Production
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.RoundByRound
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SourceMembership

/-! Π_CCS on one source as a public-coin game (SPEC.md, Algorithm 9, step 5, and Algorithm 10,
step 7), with F′'s Π_CCS model (`Spec.Folding.PiCCS.PaperJoint`).

The prover has fixed the statement and the witness, which the table `W` holds before the run. The
verifier draws `α`, then `γ`, then for each of the `ℓ` rounds it takes the prover's round
polynomial and draws the round challenge, and then it takes the output claims (`game`). Each
draw has the bad set of `RoundByRound`: some coordinate of `α` hits its set, `γ` hits its set, or
a round challenge hits its set. These sets have the densities `ℓ / |K|`, `(J − 1) / |K|` and
`width / |K|` (`game_densityLe`, with `ccsError` above each). On each path that avoids them, an
accepted run whose output claims hold gives the input claim (`game_avoids`), by
`RoundByRound.falseAcceptance_splits`. -/

namespace NightstreamFPrime.Spec.Compression.Fresh

open Finset NightstreamFPrime.Spec NightstreamFPrime.Spec.FieldExtensionRing
open NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.RoundByRound
open NightstreamFPrime.Spec.Compression.Game

open scoped BigOperators

attribute [local instance low] Classical.propDecidable

section Counting

variable {C : Type} [Fintype C]

/-- The elements of `C` with the property `p`, decided classically. It is irreducible, so that no
check of a definitional equality lists the elements of `C`. -/
@[irreducible] noncomputable def chosen (p : C → Prop) : Finset C := by
  classical
  exact univ.filter p

theorem mem_chosen {p : C → Prop} {x : C} : x ∈ chosen p ↔ p x := by
  classical
  unfold chosen
  simp

theorem density_chosen (p : C → Prop) [Nonempty C] :
    ((chosen p).card : ℝ) / Fintype.card C = 𝔼 x, if p x then (1 : ℝ) else 0 := by
  rw [Finset.expect_eq_sum_div_card, Finset.sum_boole, card_univ]
  unfold chosen
  rfl

/-- When the set `S a` does not depend on coordinate `i` of `a` and has at most one element, at
most `|C|^(n − 1)` points have coordinate `i` in their set. -/
theorem card_coordinate_le {n : ℕ} (i : Fin n) (S : (Fin n → C) → Finset C)
    (local_ : ∀ a v, S (Function.update a i v) = S a) (small : ∀ a, (S a).card ≤ 1) :
    (chosen fun a => a i ∈ S a).card ≤ Fintype.card C ^ (n - 1) := by
  have rest : Fintype.card {j : Fin n // j ≠ i} = n - 1 := by
    rw [Fintype.card_subtype_compl, Fintype.card_fin, Fintype.card_subtype_eq]
  rw [← rest, ← Fintype.card_fun]
  refine Finset.card_le_card_of_injOn (fun a (j : {j : Fin n // j ≠ i}) => a j)
    (fun _ _ => Finset.mem_coe.mpr (mem_univ _)) ?_
  intro a ha b hb same
  replace ha := mem_chosen.mp (mem_coe.mp ha)
  replace hb := mem_chosen.mp (mem_coe.mp hb)
  have others : ∀ j, j ≠ i → a j = b j := fun j hj => congrFun same ⟨j, hj⟩
  have shifted : b = Function.update a i (b i) := by
    funext j
    by_cases hj : j = i
    · subst hj
      simp
    · rw [Function.update_of_ne hj, others j hj]
  have sets : S b = S a := by rw [shifted, local_]
  rw [sets] at hb
  have at_i : a i = b i := Finset.card_le_one.mp (small a) _ ha _ hb
  funext j
  by_cases hj : j = i
  · subst hj
    exact at_i
  · exact others j hj

theorem card_exists_le {n : ℕ} (P : Fin n → (Fin n → C) → Prop) (bound : ℕ)
    (each : ∀ i, (chosen (P i)).card ≤ bound) :
    (chosen fun a => ∃ i, P i a).card ≤ n * bound := by
  calc (chosen fun a => ∃ i, P i a).card
      ≤ (univ.biUnion fun i => chosen (P i)).card := by
        refine Finset.card_le_card fun a member => ?_
        obtain ⟨i, holds⟩ := mem_chosen.mp member
        exact mem_biUnion.mpr ⟨i, mem_univ _, mem_chosen.mpr holds⟩
    _ ≤ ∑ i, (chosen (P i)).card := Finset.card_biUnion_le
    _ ≤ ∑ _i : Fin n, bound := Finset.sum_le_sum fun i _ => each i
    _ = n * bound := by simp

end Counting

section Game

variable {Commitment PublicInput : Type} {shape : Shape} {columns blockCount : ℕ}
  (statement : Statement K Commitment PublicInput shape columns blockCount baseOps)
  (witness : OutputWitness shape columns) (width : ℕ)

/-- The claimed protocol data of the run on `witness`. -/
noncomputable abbrev claimed : ProtocolPolynomial.Data K shape :=
  statement.sourceProtocolData K.embed witness

/-- The point with the coordinates `a`. -/
def pointOf (a : Fin shape.cubeVariables → K) : CubePoint K shape.cubeVariables :=
  ⟨List.ofFn a, List.length_ofFn⟩

/-- The `α` with some coordinate in its set of `RoundByRound.alphaBad`. -/
noncomputable def alphaBadSet : Finset (Fin shape.cubeVariables → K) :=
  chosen fun a => ∃ i, a i ∈
    alphaBad ((claimed statement witness).toJointData extensionOps) (pointOf a) i

/-- The `γ` of `RoundByRound.gammaBad`. -/
noncomputable def gammaBadSet (a : Fin shape.cubeVariables → K) : Finset K :=
  chosen fun g => g ∈ gammaBad ((claimed statement witness).toJointData extensionOps)
    (pointOf a)

/-- The challenges of a round of `RoundByRound.roundBad`, after the challenges `points` and with
the round polynomial `message` and `remaining` rounds after it. -/
noncomputable def roundBadSet (a : Fin shape.cubeVariables → K) (g : K) (points : List K)
    (message : FixedPolynomial K width) (remaining : ℕ) : Finset K :=
  chosen fun r => r ∈ roundBad (claimed statement witness) (pointOf a) g points message remaining

instance fixedPolynomial_nonempty : Nonempty (FixedPolynomial K width) :=
  ⟨⟨List.replicate (width + 1) K.zero, List.length_replicate⟩⟩

instance fullOutput_nonempty : Nonempty (FullOutputCoordinates.FullOutput K shape) :=
  ⟨⟨fun _ _ => K.zero, fun _ _ _ => K.zero⟩⟩

/-- The rounds: for each round, the prover's round polynomial, then the round challenge. -/
noncomputable def roundGame (a : Fin shape.cubeVariables → K) (g : K) :
    ℕ → List K → List (FixedPolynomial K width) →
      Game (List K × List (FixedPolynomial K width))
  | 0, points, messages => .pure (points, messages)
  | n + 1, points, messages => .recv (FixedPolynomial K width) fun message =>
      .draw K (roundBadSet statement witness width a g points message n) fun r =>
        roundGame a g n (points ++ [r]) (messages ++ [message])

/-- The probe of a run with the challenges `a`, `g`, `points`, the round polynomials `messages`
and the output claims `output`. -/
def probeOf (a : Fin shape.cubeVariables → K) (g : K) (points : List K)
    (length : points.length = shape.cubeVariables) (messages : List (FixedPolynomial K width))
    (output : FullOutputCoordinates.FullOutput K shape) : Probe K shape :=
  ⟨⟨pointOf a, g, ⟨points, length⟩⟩, ⟨FixedPhase.RawCertificate.encode ⟨messages⟩, output⟩⟩

/-- The probe of the run, when it has `ℓ` round challenges and the verifier accepts it. -/
noncomputable def finish (a : Fin shape.cubeVariables → K) (g : K)
    (run : List K × List (FixedPolynomial K width))
    (output : FullOutputCoordinates.FullOutput K shape) : Option (Probe K shape) :=
  if length : run.1.length = shape.cubeVariables then
    if (probeOf width a g run.1 length run.2 output).FixedWidthAccepted extensionOps K.embed
        statement width then some (probeOf width a g run.1 length run.2 output)
    else none
  else none

theorem finish_eq_some {a : Fin shape.cubeVariables → K} {g : K}
    {run : List K × List (FixedPolynomial K width)}
    {output : FullOutputCoordinates.FullOutput K shape} {probe : Probe K shape}
    (equal : finish statement width a g run output = some probe) :
    ∃ length, probe = probeOf width a g run.1 length run.2 output ∧
      probe.FixedWidthAccepted extensionOps K.embed statement width := by
  unfold finish at equal
  split_ifs at equal with length accepted
  cases equal
  exact ⟨length, rfl, accepted⟩

/-- Π_CCS on `statement` against an adaptive prover: the probe of an accepted run. -/
noncomputable def game : Game (Option (Probe K shape)) :=
  .draw (Fin shape.cubeVariables → K) (alphaBadSet statement witness) fun a =>
    .draw K (gammaBadSet statement witness a) fun g =>
      Game.bind (roundGame statement witness width a g shape.cubeVariables [] []) fun run =>
        .recv (FullOutputCoordinates.FullOutput K shape) fun output =>
          .pure (finish statement width a g run output)

/-- The round challenges avoid their sets. -/
def RoundsGood (a : Fin shape.cubeVariables → K) (g : K) (points : List K)
    (messages : List (FixedPolynomial K width)) : Prop :=
  ∀ j (hp : j < points.length) (hm : j < messages.length),
    points[j] ∉ roundBad (claimed statement witness) (pointOf a) g (points.take j) messages[j]
      (shape.cubeVariables - j - 1)

theorem roundGame_avoids (a : Fin shape.cubeVariables → K) (g : K) :
    ∀ (n : ℕ) (points : List K) (messages : List (FixedPolynomial K width)),
      points.length + n = shape.cubeVariables → messages.length = points.length →
      RoundsGood statement witness width a g points messages →
      (roundGame statement witness width a g n points messages).Avoids fun run =>
        run.1.length = shape.cubeVariables ∧ run.2.length = shape.cubeVariables ∧
          RoundsGood statement witness width a g run.1 run.2
  | 0, points, messages, length, same, good =>
      ⟨show points.length = _ by omega, show messages.length = _ by omega, good⟩
  | n + 1, points, messages, length, same, good => by
      intro message r outside
      refine roundGame_avoids a g n _ _
        (by rw [List.length_append, List.length_singleton]; omega)
        (by simp [same]) ?_
      intro j hp hm
      simp only [List.length_append, List.length_singleton] at hp hm
      by_cases last : j = points.length
      · subst last
        have notIn := mem_chosen.not.mp outside
        have point : (points ++ [r])[points.length]'(by simp) = r := by simp
        have prefix_ : (points ++ [r]).take points.length = points := by simp
        have round : (messages ++ [message])[points.length]'(by simp [same]) = message := by
          simp [List.getElem_append_right, same]
        rw [point, prefix_, round, show shape.cubeVariables - points.length - 1 = n by omega]
        exact notIn
      · have below : j < points.length := by omega
        have belowMessages : j < messages.length := by omega
        rw [List.getElem_append_left below, List.getElem_append_left belowMessages,
          List.take_append_of_le_length below.le]
        exact good j below belowMessages

/-- On each path that avoids the bad sets, an accepted run whose output claims hold gives the
input claims. -/
theorem game_avoids {openingMaps : OpeningMaps Commitment PublicInput columns}
    {params : GlobalParams} (freshBound : params.b = 2)
    (constantLaw : MatrixCoefficientSource.ConstantTermLaw baseOps statement.matrixSource.kernel)
    (degreeCovers : (statement.verifierInput K.embed).sumcheckDegreeBound ≤ width) :
    (game statement witness width).Avoids fun out => ∀ probe, out = some probe →
      AmbientOutputHolds extensionOps K.embed openingMaps params statement probe witness →
        SourceHolds extensionOps K.embed openingMaps params statement witness := by
  intro a alphaGood g gammaGood
  rw [avoids_bind]
  refine (roundGame_avoids statement witness width a g _ [] [] (by simp) rfl
    (fun j hp => absurd hp (by simp))).mono fun run ⟨length, messagesLength, good⟩ => ?_
  intro output probe equal ambient
  obtain ⟨_, rfl, accepted⟩ := finish_eq_some statement width equal
  by_contra invalid
  rcases falseAcceptance_splits openingMaps params freshBound statement constantLaw
      degreeCovers witness invalid _ ambient accepted with
    ⟨index, member⟩ | member | ⟨certificate, round, message, decoded, located, member⟩
  · apply alphaGood
    refine mem_chosen.mpr ⟨index, ?_⟩
    have coordinateEq : coordinate (pointOf a) index = a index := List.getElem_ofFn ..
    rwa [show (probeOf width a g run.1 _ run.2 output).coins.alpha = pointOf a from rfl,
      coordinateEq] at member
  · rw [show (probeOf width a g run.1 _ run.2 output).coins.gamma = g from rfl,
      show (probeOf width a g run.1 _ run.2 output).coins.alpha = pointOf a from rfl] at member
    have member' : g ∈ gammaBad ((claimed statement witness).toJointData extensionOps)
        (pointOf a) := member
    apply gammaGood
    refine mem_chosen.mpr ?_
    exact member'
  · simp only [probeOf, FixedPhase.RawCertificate.decode_encode, Option.some.injEq] at decoded
    subst decoded
    have roundLt : round.val < run.1.length := by rw [length]; exact round.isLt
    have messageLt : round.val < run.2.length := by rw [messagesLength]; exact round.isLt
    have equal : run.2[round.val] = message := by
      simpa [List.getElem?_eq_getElem messageLt] using located
    apply good round.val roundLt messageLt
    rw [equal]
    exact member

/-- The density of each bad set of the run is at most `ccsError`. -/
noncomputable def ccsError (shape : Shape) (width : ℕ) : ℝ :=
  ((shape.cubeVariables + shape.jointCoefficientCount + width : ℕ) : ℝ) / Fintype.card K

theorem card_K_pos : (0 : ℝ) < Fintype.card K := by exact_mod_cast Fintype.card_pos

theorem alpha_density :
    ((alphaBadSet statement witness).card : ℝ) /
        Fintype.card (Fin shape.cubeVariables → K) ≤
      shape.cubeVariables / Fintype.card K := by
  let S : Fin shape.cubeVariables → (Fin shape.cubeVariables → K) → Finset K := fun i a =>
    chosen (· ∈ alphaBad ((claimed statement witness).toJointData extensionOps) (pointOf a) i)
  have each : ∀ i, (chosen fun a => a i ∈ S i a).card ≤
      Fintype.card K ^ (shape.cubeVariables - 1) := by
    intro i
    refine card_coordinate_le i (S i) (fun a v => ?_) (fun a => ?_)
    · have point : pointOf (Function.update a i v) =
          ⟨(pointOf a).coordinates.set i.val v, by simp [pointOf]⟩ := by
        simp only [pointOf, CubePoint.mk.injEq]
        apply List.ext_getElem (by simp)
        intro k h₁ h₂
        simp only [List.getElem_ofFn, List.getElem_set]
        by_cases same : i.val = k
        · subst same
          simp
        · rw [ite_eq_right same, Function.update_of_ne (fun e => same (congrArg Fin.val e).symm)]
      simp only [S]
      rw [point, alphaBad_set]
    · have bound := alphaBad_probability_le univ
        ((claimed statement witness).toJointData extensionOps) (pointOf a) i
      rw [← density_chosen (fun v => v ∈ alphaBad
        ((claimed statement witness).toJointData extensionOps) (pointOf a) i), card_univ] at bound
      have cardLe : ((S i a).card : ℝ) ≤ 1 := (div_le_div_iff_of_pos_right card_K_pos).mp bound
      exact_mod_cast cardLe
  have total := card_exists_le _ _ each
  have same : alphaBadSet statement witness = chosen fun a => ∃ i, a i ∈ S i a :=
    congrArg chosen (funext fun a => propext (exists_congr fun i => mem_chosen.symm))
  have cards : (Fintype.card (Fin shape.cubeVariables → K) : ℝ) =
      Fintype.card K ^ shape.cubeVariables := by
    rw [Fintype.card_fun, Fintype.card_fin, Nat.cast_pow]
  rw [same]
  rcases Nat.eq_zero_or_pos shape.cubeVariables with zero | positive
  · have empty : (chosen fun a : Fin shape.cubeVariables → K => ∃ i, a i ∈ S i a) = ∅ :=
      Finset.eq_empty_of_forall_notMem fun a member => by
        obtain ⟨i, _⟩ := mem_chosen.mp member
        exact absurd i.isLt (by omega)
    rw [empty, Finset.card_empty, Nat.cast_zero, zero_div]
    positivity
  · rw [cards, div_le_div_iff₀ (by positivity) card_K_pos]
    have total' : ((chosen fun a => ∃ i, a i ∈ S i a).card : ℝ) ≤
        shape.cubeVariables * (Fintype.card K : ℝ) ^ (shape.cubeVariables - 1) := by
      exact_mod_cast total
    calc ((chosen fun a => ∃ i, a i ∈ S i a).card : ℝ) * Fintype.card K
        ≤ shape.cubeVariables * (Fintype.card K : ℝ) ^ (shape.cubeVariables - 1) *
            Fintype.card K := mul_le_mul_of_nonneg_right total' card_K_pos.le
      _ = shape.cubeVariables * Fintype.card K ^ shape.cubeVariables := by
          rw [mul_assoc, ← pow_succ, Nat.sub_add_cancel positive]

theorem gamma_density (a : Fin shape.cubeVariables → K) :
    ((gammaBadSet statement witness a).card : ℝ) / Fintype.card K ≤
      (shape.jointCoefficientCount - 1 : ℕ) / Fintype.card K := by
  have bound := gammaBad_probability_le univ ((claimed statement witness).toJointData extensionOps)
    (pointOf a)
  rw [card_univ] at bound
  rw [gammaBadSet, density_chosen]
  exact bound

theorem round_density (degreeCovers : (statement.verifierInput K.embed).sumcheckDegreeBound ≤ width)
    (a : Fin shape.cubeVariables → K) (g : K) (points : List K) (message : FixedPolynomial K width)
    (remaining : ℕ) (length : points.length + 1 + remaining = shape.cubeVariables) :
    ((roundBadSet statement witness width a g points message remaining).card : ℝ) /
        Fintype.card K ≤ width / Fintype.card K := by
  have bound := roundBad_probability_le univ (claimed statement witness) (pointOf a) g
    degreeCovers points message remaining length
  rw [card_univ] at bound
  rw [roundBadSet, density_chosen]
  exact bound

theorem roundGame_densityLe (degreeCovers :
    (statement.verifierInput K.embed).sumcheckDegreeBound ≤ width)
    (a : Fin shape.cubeVariables → K) (g : K) :
    ∀ (n : ℕ) (points : List K) (messages : List (FixedPolynomial K width)),
      points.length + n = shape.cubeVariables →
      (roundGame statement witness width a g n points messages).DensityLe
        (width / Fintype.card K)
  | 0, _, _, _ => trivial
  | n + 1, points, messages, length => fun message =>
      ⟨round_density statement witness width degreeCovers a g points message n (by omega),
        fun r => roundGame_densityLe degreeCovers a g n _ _ (by simp; omega)⟩

theorem game_densityLe (degreeCovers :
    (statement.verifierInput K.embed).sumcheckDegreeBound ≤ width) :
    (game statement witness width).DensityLe (ccsError shape width) := by
  have scale (m : ℕ) (le : m ≤ shape.cubeVariables + shape.jointCoefficientCount + width) :
      (m : ℝ) / Fintype.card K ≤ ccsError shape width :=
    div_le_div_of_nonneg_right (by exact_mod_cast le) card_K_pos.le
  refine ⟨(alpha_density statement witness).trans (scale _ (by omega)), fun a =>
    ⟨(gamma_density statement witness a).trans (scale _ (by omega)), fun g => ?_⟩⟩
  exact densityLe_bind (P := fun _ => True)
    ((roundGame_densityLe statement witness width degreeCovers a g _ [] [] (by simp)).mono
      (scale _ (by omega))) (Ends.of_forall (fun _ => trivial) _) fun _ _ _ => trivial

/-- In the state-restoration experiment with `q` queries, a prover makes an accepted run whose
output claims hold, without the input claims, with probability at most `q · ccsError`. -/
theorem soundness {openingMaps : OpeningMaps Commitment PublicInput columns}
    {params : GlobalParams} (freshBound : params.b = 2)
    (constantLaw : MatrixCoefficientSource.ConstantTermLaw baseOps statement.matrixSource.kernel)
    (degreeCovers : (statement.verifierInput K.embed).sumcheckDegreeBound ≤ width)
    (s : Strategy (Option (Option (Probe K shape)))) (q : ℕ) :
    prob s (restore (game statement witness width) q []) (fun r => ¬ ∀ out, r = some out →
      ∀ probe, out = some probe →
        AmbientOutputHolds extensionOps K.embed openingMaps params statement probe witness →
          SourceHolds extensionOps K.embed openingMaps params statement witness) ≤
      q * ccsError shape width :=
  restore_sound _ s _ _ (game_avoids statement witness width freshBound constantLaw degreeCovers)
    (game_densityLe statement witness width degreeCovers) (by unfold ccsError; positivity) q

end Game

end NightstreamFPrime.Spec.Compression.Fresh

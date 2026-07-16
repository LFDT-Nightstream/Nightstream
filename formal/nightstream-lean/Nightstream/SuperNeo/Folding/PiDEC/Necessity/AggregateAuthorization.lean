import Nightstream.SuperNeo.Folding.PiDEC.Necessity.DigitAuthorization

/-!
Necessity of pointwise `Pi_DEC` child authorization.

Owns: fixed-arity counterexamples showing that aggregate digit sums and
aggregate norm sums do not determine which child carries a value, plus a
weakened aggregate-only authorization that can feed different next-stage child
vectors.

Does not own: the production child carrier, the full `Pi_DEC` verifier,
Goldilocks encoding, transcripts, Rust/R1CS refinement, row removal, or
constraint counts.

Emits constraints: no.

Authority boundary: the witness uses the binary subset `{0,1}` only because it
is contained in the production centered alphabet. It does not claim binary is
the protocol alphabet. Fixed arity is intrinsic (`Fin 14`), so the ambiguity is
solely loss of pointwise child identity.

| Protocol | Phase | Omitted family | Kernel-checked invalid ambiguity |
|---|---|---|---|
| `Pi_DEC` | child values | pointwise digit identity | the same digit sum places one on child zero or child one |
| `Pi_DEC` | child norms | pointwise norm identity | the same norm sum places one on child zero or child one |
| NIFS | next input | exact checked-child wiring | aggregate-only authorization feeds two different recompositions |
-/

namespace Nightstream.SuperNeo.Folding.PiDEC.Necessity.AggregateAuthorization

abbrev ChildVector := Fin 14 -> Nat

/-- One at child zero and zero elsewhere. -/
def firstChildHot : ChildVector := fun child =>
  if child.val = 0 then 1 else 0

/-- One at child one and zero elsewhere. -/
def secondChildHot : ChildVector := fun child =>
  if child.val = 1 then 1 else 0

/-- The deliberately weak aggregate that forgets child position. -/
def aggregateDigitSum (children : ChildVector) : Nat :=
  (List.ofFn children).sum

/-- Exact radix-two recomposition, which retains child position. -/
def recompose (children : ChildVector) : Nat :=
  (List.ofFn fun child => 2 ^ child.val * children child).sum

def BinarySubset (children : ChildVector) : Prop :=
  forall child, children child < 2

theorem firstChildHot_binary : BinarySubset firstChildHot := by
  intro child
  unfold firstChildHot
  split <;> omega

theorem secondChildHot_binary : BinarySubset secondChildHot := by
  intro child
  unfold secondChildHot
  split <;> omega

theorem firstChildHot_sum : aggregateDigitSum firstChildHot = 1 := by
  decide

theorem secondChildHot_sum : aggregateDigitSum secondChildHot = 1 := by
  decide

theorem firstChildHot_recompose : recompose firstChildHot = 1 := by
  decide

theorem secondChildHot_recompose : recompose secondChildHot = 2 := by
  decide

theorem firstChildHot_ne_secondChildHot :
    firstChildHot ≠ secondChildHot := by
  intro equal
  have atZero := congrFun equal (show Fin 14 from ⟨0, by decide⟩)
  simp [firstChildHot, secondChildHot] at atZero

/-- Aggregate digit sums are not functional even at exact child arity and in
the binary subset of the centered alphabet. -/
theorem aggregate_digit_sum_not_functional_for_fixed_child_count :
    ¬ (forall left right : ChildVector,
      BinarySubset left -> BinarySubset right ->
      aggregateDigitSum left = aggregateDigitSum right -> left = right) := by
  intro functional
  apply firstChildHot_ne_secondChildHot
  exact functional firstChildHot secondChildHot
    firstChildHot_binary secondChildHot_binary
    (firstChildHot_sum.trans secondChildHot_sum.symm)

/-- The same positional ambiguity applies to a total norm summary. -/
theorem aggregate_norm_sum_not_functional_for_fixed_child_count :
    ¬ (forall left right : ChildVector,
      aggregateDigitSum left = aggregateDigitSum right -> left = right) := by
  intro functional
  apply firstChildHot_ne_secondChildHot
  exact functional firstChildHot secondChildHot
    (firstChildHot_sum.trans secondChildHot_sum.symm)

/-- Aggregate-only validation: exact arity and per-child binary range are kept,
but positional recomposition is omitted. -/
def AggregateOnlyChildValidation
    (summary : Nat) (children : ChildVector) : Prop :=
  BinarySubset children /\ aggregateDigitSum children = summary

/-- The next input is wired to the table accepted by the weakened validator.
This keeps wire identity while isolating the weakness to aggregate validation. -/
structure AcceptedAggregateOnlyChildVector
    (summary : Nat) (children nextInput : ChildVector) : Prop where
  proofVerified : AggregateOnlyChildValidation summary children
  wireIdentity : nextInput = children

/-- The same aggregate summary authorizes different next inputs with different
radix-two recompositions. -/
theorem aggregate_only_validation_can_feed_different_next_inputs :
    AcceptedAggregateOnlyChildVector 1 firstChildHot firstChildHot /\
      AcceptedAggregateOnlyChildVector 1 secondChildHot secondChildHot /\
      aggregateDigitSum firstChildHot = aggregateDigitSum secondChildHot /\
      firstChildHot ≠ secondChildHot /\
      recompose firstChildHot ≠ recompose secondChildHot := by
  refine ⟨⟨⟨firstChildHot_binary, firstChildHot_sum⟩, rfl⟩,
    ⟨⟨secondChildHot_binary, secondChildHot_sum⟩, rfl⟩,
    firstChildHot_sum.trans secondChildHot_sum.symm,
    firstChildHot_ne_secondChildHot, ?_⟩
  rw [firstChildHot_recompose, secondChildHot_recompose]
  decide

end Nightstream.SuperNeo.Folding.PiDEC.Necessity.AggregateAuthorization

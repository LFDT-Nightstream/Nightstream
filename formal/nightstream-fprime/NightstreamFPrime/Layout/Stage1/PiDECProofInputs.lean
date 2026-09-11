import NightstreamFPrime.Layout.Stage1.PiDECInputBounds

/-!
Owns loading the canonical PiDEC source fields from the actual proof messages
and the verifier's public-input split. The four ranges and their typed coordinate
maps come from the existing PiDEC owners. This loader adds no circuit rows.
-/

namespace NightstreamFPrime.Layout.Stage1.PiDECProofInputs

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiDEC.v1_1
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open PiDECInputs

private def writeFamily {count width : Nat} (env : Env) (start : Nat)
    (values : Fin count → Fin width → F) : Env := fun index =>
  if inside : start ≤ index ∧ index < start + count * width then
    let packed : Fin (count * width) := ⟨index - start, by omega⟩
    let coordinates := finProdFinEquiv.symm packed
    values coordinates.1 coordinates.2
  else env index

private theorem writeFamily_read {count width : Nat} (env : Env) (start : Nat)
    (values : Fin count → Fin width → F) (child : Fin count) (coordinate : Fin width) :
    writeFamily env start values (start + (finProdFinEquiv (child, coordinate)).val) =
      values child coordinate := by
  have inside : start ≤ start + (finProdFinEquiv (child, coordinate)).val ∧
      start + (finProdFinEquiv (child, coordinate)).val < start + count * width :=
    ⟨Nat.le_add_right _ _, Nat.add_lt_add_left (finProdFinEquiv (child, coordinate)).isLt start⟩
  rw [writeFamily, dif_pos inside]
  have packedEq :
      (⟨start + (finProdFinEquiv (child, coordinate)).val - start, by omega⟩ : Fin (count * width)) =
      finProdFinEquiv (child, coordinate) := by
    apply Fin.ext
    exact Nat.add_sub_cancel_left _ _
  simp only [packedEq, Equiv.symm_apply_apply]

private theorem writeFamily_agreesOutside {count width : Nat} (env : Env) (start : Nat)
    (values : Fin count → Fin width → F) :
    AgreesOutside env (writeFamily env start values) start (count * width) := by
  intro index outside
  have notInside : ¬ (start ≤ index ∧ index < start + count * width) := by omega
  rw [writeFamily, dif_neg notInside]

private theorem writeFamily_before {count width : Nat} (env : Env) (start : Nat)
    (values : Fin count → Fin width → F) (index : Nat) (below : index < start) :
    writeFamily env start values index = env index :=
  writeFamily_agreesOutside env start values index (Or.inl below)

private def commitmentWord {degree : Nat} (proof : Proof degree)
    (child : Fin childCount) (coordinate : Fin commitmentWordsPerChild) : F :=
  let location := CommitmentRecomposition.coordinates coordinate
  proof.piDecCommitments child location.1 location.2

private def evalKWord {degree : Nat} (proof : Proof degree)
    (child : Fin childCount) (coordinate : Fin evalKWordsPerChild) : F :=
  let location := RingKRecomposition.coordinates coordinate
  RingKRecomposition.kCell location.2.2 ((proof.piDecEvaluations child).pad location.2.1)

private def evalAWord {degree : Nat} (proof : Proof degree)
    (child : Fin childCount) (coordinate : Fin evalAWordsPerChild) : F :=
  let location := RingKRecomposition.coordinates coordinate
  RingKRecomposition.kCell location.2.2 ((proof.piDecEvaluations child).matrix location.1 location.2.1)

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

/-- Load the actual prover messages and the verifier-computed public digits
into the four existing D source ranges. All other source values are retained. -/
def load {degree : Nat} (env : Env) (proof : Proof degree)
    (parent : PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits)) : Env :=
  writeFamily
    (writeFamily
      (writeFamily
        (writeFamily env commitmentInputStart (commitmentWord proof))
        evalKInputStart (evalKWord proof))
      evalAInputStart (evalAWord proof))
    publicInputStart (Phi81Relation.PiDECAlgebra.PublicInput.splitPublicInput parent)

/-- Loading D input fields preserves every source outside the existing
proof-input interval, including all earlier C/R witness and transcript values. -/
theorem load_agreesOutside {degree : Nat} (env : Env) (proof : Proof degree)
    (parent : PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    AgreesOutside env (load env proof parent) proofInputStart proofInputColumnCount := by
  have first := writeFamily_agreesOutside env commitmentInputStart (commitmentWord proof)
  have second := writeFamily_agreesOutside
    (writeFamily env commitmentInputStart (commitmentWord proof)) evalKInputStart (evalKWord proof)
  have third := writeFamily_agreesOutside
    (writeFamily (writeFamily env commitmentInputStart (commitmentWord proof))
      evalKInputStart (evalKWord proof)) evalAInputStart (evalAWord proof)
  have fourth := writeFamily_agreesOutside
    (writeFamily (writeFamily (writeFamily env commitmentInputStart (commitmentWord proof))
      evalKInputStart (evalKWord proof)) evalAInputStart (evalAWord proof))
    publicInputStart (Phi81Relation.PiDECAlgebra.PublicInput.splitPublicInput parent)
  intro index outside
  have outsideAll : index < commitmentInputStart ∨
      commitmentInputStart + childCount * commitmentWordsPerChild + childCount * evalKWordsPerChild +
        childCount * evalAWordsPerChild + childCount * publicInputWordsPerChild ≤ index := by
    simpa only [commitmentInputStart, proofInputColumnCount, Nat.mul_add, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm, Nat.mul_comm] using outside
  have outsideK : index < evalKInputStart ∨
      evalKInputStart + childCount * evalKWordsPerChild ≤ index := by
    unfold evalKInputStart
    omega
  have outsideA : index < evalAInputStart ∨
      evalAInputStart + childCount * evalAWordsPerChild ≤ index := by
    unfold evalAInputStart evalKInputStart
    omega
  have outsidePublic : index < publicInputStart ∨
      publicInputStart + childCount * publicInputWordsPerChild ≤ index := by
    unfold publicInputStart evalAInputStart evalKInputStart
    omega
  exact (fourth index outsidePublic).trans ((third index outsideA).trans
    ((second index outsideK).trans (first index (by omega))))

private theorem load_commitmentWord {degree : Nat} (env : Env) (proof : Proof degree)
    (parent : PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    (child : Fin childCount) (coordinate : Fin commitmentWordsPerChild) :
    load env proof parent (commitmentInputStart + (finProdFinEquiv (child, coordinate)).val) =
      commitmentWord proof child coordinate := by
  let index := commitmentInputStart + (finProdFinEquiv (child, coordinate)).val
  have beforeK : index < evalKInputStart :=
    Nat.add_lt_add_left (finProdFinEquiv (child, coordinate)).isLt commitmentInputStart
  have beforeA : index < evalAInputStart := by unfold evalAInputStart; omega
  have beforePublic : index < publicInputStart := by unfold publicInputStart; omega
  change load env proof parent index = _
  rw [load, writeFamily_before _ publicInputStart _ _ beforePublic,
    writeFamily_before _ evalAInputStart _ _ beforeA,
    writeFamily_before _ evalKInputStart _ _ beforeK]
  exact writeFamily_read env commitmentInputStart (commitmentWord proof) child coordinate

private theorem load_evalKWord {degree : Nat} (env : Env) (proof : Proof degree)
    (parent : PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    (child : Fin childCount) (coordinate : Fin evalKWordsPerChild) :
    load env proof parent (evalKInputStart + (finProdFinEquiv (child, coordinate)).val) =
      evalKWord proof child coordinate := by
  let index := evalKInputStart + (finProdFinEquiv (child, coordinate)).val
  have beforeA : index < evalAInputStart :=
    Nat.add_lt_add_left (finProdFinEquiv (child, coordinate)).isLt evalKInputStart
  have beforePublic : index < publicInputStart := by unfold publicInputStart; omega
  change load env proof parent index = _
  rw [load, writeFamily_before _ publicInputStart _ _ beforePublic,
    writeFamily_before _ evalAInputStart _ _ beforeA]
  exact writeFamily_read _ evalKInputStart (evalKWord proof) child coordinate

private theorem load_evalAWord {degree : Nat} (env : Env) (proof : Proof degree)
    (parent : PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    (child : Fin childCount) (coordinate : Fin evalAWordsPerChild) :
    load env proof parent (evalAInputStart + (finProdFinEquiv (child, coordinate)).val) =
      evalAWord proof child coordinate := by
  have beforePublic : evalAInputStart + (finProdFinEquiv (child, coordinate)).val < publicInputStart :=
    Nat.add_lt_add_left (finProdFinEquiv (child, coordinate)).isLt evalAInputStart
  rw [load, writeFamily_before _ publicInputStart _ _ beforePublic]
  exact writeFamily_read _ evalAInputStart (evalAWord proof) child coordinate

private theorem load_publicWord {degree : Nat} (env : Env) (proof : Proof degree)
    (parent : PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    (child : Fin childCount) (coordinate : Fin publicInputWordsPerChild) :
    load env proof parent (publicInputStart + (finProdFinEquiv (child, coordinate)).val) =
      Phi81Relation.PiDECAlgebra.PublicInput.splitPublicInput parent child coordinate := by
  exact writeFamily_read _ publicInputStart
    (Phi81Relation.PiDECAlgebra.PublicInput.splitPublicInput parent) child coordinate

private theorem k_ext (left right : K)
    (c0 : left.c0 = right.c0) (c1 : left.c1 = right.c1) : left = right := by
  cases left
  cases right
  simp_all

/-- Every loaded commitment coefficient is the actual D proof coefficient. -/
theorem eval_childCommitment {degree : Nat} (env : Env) (proof : Proof degree)
    (parent : PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    (child : Phi81Relation.PiDECAlgebra.Radix.ChildIndex)
    (row : Fin productionProfile.commitmentWidth) (lane : Fin ringDegree) :
    (childCommitment child row lane).eval (load env proof parent) = proof.piDecCommitments child row lane := by
  have word := load_commitmentWord env proof parent child (CommitmentRecomposition.indexOf row lane)
  simp only [commitmentWord, CommitmentRecomposition.coordinates_indexOf] at word
  convert word using 1 <;> simp only [childCommitment, childCommitmentStart, Expr.eval,
    CommitmentRecomposition.indexOf, finProdFinEquiv, Equiv.coe_fn_mk, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm, Nat.mul_comm]

/-- Every loaded Pad coefficient is the actual D proof coefficient. -/
theorem eval_childEvalK {degree : Nat} (env : Env) (proof : Proof degree)
    (parent : PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    (child : Phi81Relation.PiDECAlgebra.Radix.ChildIndex)
    (coefficient : Fin productionShape.coefficientCount) :
    (childEvalK child coefficient).eval (load env proof parent) = (proof.piDecEvaluations child).pad coefficient := by
  have cell (part : Fin RingKRecomposition.cellCount) := load_evalKWord env proof parent child
    (RingKRecomposition.indexOf EvalKRecomposition.block coefficient part)
  simp only [evalKWord, RingKRecomposition.coordinates_indexOf] at cell
  apply k_ext
  · convert cell RingKRecomposition.c0Cell using 1 <;>
      simp only [childEvalK, childEvalKStart, Quadratic.KExpr.eval, Expr.eval,
        RingKRecomposition.indexOf, EvalKRecomposition.block, RingKRecomposition.c0Cell,
        RingKRecomposition.kCell, finProdFinEquiv, Equiv.coe_fn_mk, Nat.zero_mul,
        Nat.zero_add, Nat.add_zero, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm, Nat.mul_comm, if_pos rfl]
  · convert cell RingKRecomposition.c1Cell using 1 <;>
      simp only [childEvalK, childEvalKStart, Quadratic.KExpr.eval, Expr.eval,
        RingKRecomposition.indexOf, EvalKRecomposition.block, RingKRecomposition.c1Cell,
        RingKRecomposition.kCell, finProdFinEquiv, Equiv.coe_fn_mk, Nat.zero_mul,
        Nat.zero_add, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm, Nat.mul_comm, if_neg (by decide : ¬ (1 : Nat) = 0)]

/-- Every loaded matrix-evaluation coefficient is the actual D proof coefficient. -/
theorem eval_childEvalA {degree : Nat} (env : Env) (proof : Proof degree)
    (parent : PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    (child : Phi81Relation.PiDECAlgebra.Radix.ChildIndex)
    (matrix : Fin productionShape.matrixCount) (coefficient : Fin productionShape.coefficientCount) :
    (childEvalA child matrix coefficient).eval (load env proof parent) =
      (proof.piDecEvaluations child).matrix matrix coefficient := by
  have cell (part : Fin RingKRecomposition.cellCount) := load_evalAWord env proof parent child
    (RingKRecomposition.indexOf matrix coefficient part)
  simp only [evalAWord, RingKRecomposition.coordinates_indexOf] at cell
  apply k_ext
  · convert cell RingKRecomposition.c0Cell using 1 <;>
      simp only [childEvalA, childEvalAStart, Quadratic.KExpr.eval, Expr.eval,
        RingKRecomposition.indexOf, RingKRecomposition.c0Cell, RingKRecomposition.kCell,
        finProdFinEquiv, Equiv.coe_fn_mk, Nat.add_zero, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm, Nat.mul_comm, if_pos rfl]
  · convert cell RingKRecomposition.c1Cell using 1 <;>
      simp only [childEvalA, childEvalAStart, Quadratic.KExpr.eval, Expr.eval,
        RingKRecomposition.indexOf, RingKRecomposition.c1Cell, RingKRecomposition.kCell,
        finProdFinEquiv, Equiv.coe_fn_mk, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm, Nat.mul_comm, if_neg (by decide : ¬ (1 : Nat) = 0)]

/-- Loaded public digits are the verifier's deterministic split of the parent. -/
theorem eval_childPublicInput {degree : Nat} (env : Env) (proof : Proof degree)
    (parent : PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    (child : Phi81Relation.PiDECAlgebra.Radix.ChildIndex) (coordinate : Fin 270) :
    (childPublicInput child coordinate).eval (load env proof parent) =
      Phi81Relation.PiDECAlgebra.PublicInput.splitPublicInput parent child coordinate := by
  have word := load_publicWord env proof parent child coordinate
  convert word using 1 <;> simp only [childPublicInput, childPublicInputStart, Expr.eval,
    finProdFinEquiv, Equiv.coe_fn_mk, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm, Nat.mul_comm]

end NightstreamFPrime.Layout.Stage1.PiDECProofInputs

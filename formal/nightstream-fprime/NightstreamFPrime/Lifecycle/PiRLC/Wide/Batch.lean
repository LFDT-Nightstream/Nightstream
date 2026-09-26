import NightstreamFPrime.Lifecycle.PiRLC.Wide.Scalar
import NightstreamFPrime.Lifecycle.Types

/-! Candidate batch of 17 wide-sampler scalar calls. The outgoing state of
each call is the next call's input; centered coefficients are expression
views of the checked digits. No copy or boundary rows are added. -/

namespace NightstreamFPrime.Lifecycle.PiRLC.Wide.Batch

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Sampling

abbrev Interface := Scalar.Interface
abbrev sourceCount := productionShape.sourceCount
abbrev EState := Scalar.EState

def privateCount : Nat := sourceCount * Scalar.privateCount
def logicalRowCount : Nat := sourceCount * Scalar.logicalRowCount
def sourceOffset (offset source : Nat) : Nat := offset + source * Scalar.privateCount

def stateAtExpr (interface : Interface) (offset : Nat) : Nat → EState
  | 0 => interface.initialState offset
  | source + 1 => Scalar.outputState
      { initialState := fun _ => stateAtExpr interface offset source }
      source (sourceOffset offset source)

def childInterface (interface : Interface) (offset source : Nat) : Scalar.Interface where
  initialState := fun _ => stateAtExpr interface offset source

def childName (source : Nat) : String := "pirlc.wide.scalar_" ++ toString source

def childOp (interface : Interface) (offset source : Nat) : Op :=
  Sequence.childOp (childName source) (Scalar.circuit (childInterface interface offset source) source)
    (sourceOffset offset source)

def prefixOps (interface : Interface) (offset count : Nat) : List Op :=
  (List.range count).map (childOp interface offset)

def operations (interface : Interface) (offset : Nat) : List Op := prefixOps interface offset sourceCount

def outputChallenge (offset : Nat) (source : Fin sourceCount) : Fin ringDegree → Expr :=
  Scalar.outputChallenge (sourceOffset offset source.val)

def outputState (interface : Interface) (offset : Nat) : EState := stateAtExpr interface offset sourceCount

abbrev Assumptions := Scalar.Assumptions

structure SpecHolds (interface : Interface) (offset : Nat) (env : Env) : Prop where
  state : Scalar.evalState env (outputState interface offset) =
    Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.Transcript.stateAt
      (Scalar.evalState env (interface.initialState offset)) sourceCount
  digits : ∀ source : Fin sourceCount, ∀ position : Fin ringDegree,
    WideReduction.digitValue env
      (WideReduction.Program.coreOffset (Scalar.rangeOffset (sourceOffset offset source.val))) position.val =
        (Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.Transcript.scalarAt
          (Scalar.evalState env (interface.initialState offset)) source.val position).val

theorem sourceCount_eq : sourceCount = 17 := by
  rw [sourceCount, productionShape_sourceCount]
  rfl

theorem counts : privateCount = 54485 ∧ logicalRowCount = 31705 := by
  rw [privateCount, logicalRowCount, sourceCount_eq, Scalar.counts.1, Scalar.counts.2]
  decide

private theorem child_length (interface : Interface) (offset source : Nat) :
    (childOp interface offset source).localLength = Scalar.privateCount := by
  rw [childOp, Sequence.childOp_localLength]
  exact Scalar.localLength_eq _ _ _

private theorem child_rows (interface : Interface) (offset source : Nat) :
    (childOp interface offset source).rowCount = Scalar.logicalRowCount := rfl

theorem prefix_length (interface : Interface) (offset count : Nat) :
    localLength (prefixOps interface offset count) = count * Scalar.privateCount := by
  simp only [prefixOps, localLength, List.map_map, Function.comp_def, child_length,
    List.map_const', List.length_range, List.sum_replicate, smul_eq_mul]

theorem localLength_eq (interface : Interface) (offset : Nat) :
    localLength (operations interface offset) = privateCount := prefix_length _ _ _

theorem rowCount_eq (interface : Interface) (offset : Nat) :
    (flatConstraints (operations interface offset)).length = logicalRowCount := by
  rw [flatConstraints_length_eq_rowCount]
  simp only [operations, prefixOps, Circuit.rowCount, List.map_map, Function.comp_def, child_rows,
    List.map_const', List.length_range, List.sum_replicate, smul_eq_mul]
  rfl

theorem prefix_succ (interface : Interface) (offset count : Nat) :
    prefixOps interface offset (count + 1) =
      prefixOps interface offset count ++ [childOp interface offset count] := by
  simp only [prefixOps, List.range_succ, List.map_append, List.map_cons, List.map_nil]

theorem child_member (interface : Interface) (offset count source : Nat) (below : source < count) :
    childOp interface offset source ∈ prefixOps interface offset count := by
  exact List.mem_map.mpr ⟨source, List.mem_range.mpr below, rfl⟩

theorem state_below (interface : Interface) (offset : Nat) (inputs : Assumptions interface offset)
    (count : Nat) : ∀ lane, (stateAtExpr interface offset count lane).VarsBelow (sourceOffset offset count) := by
  induction count with
  | zero => exact inputs
  | succ count ih =>
      have below := Scalar.outputState_below (childInterface interface offset count) count
        (sourceOffset offset count) ih
      simpa only [stateAtExpr, sourceOffset, Nat.add_mul, Nat.one_mul, Nat.add_assoc] using! below

theorem child_inputs (interface : Interface) (offset source : Nat) (inputs : Assumptions interface offset) :
    Scalar.Assumptions (childInterface interface offset source) (sourceOffset offset source) :=
  state_below interface offset inputs source

theorem child_scope (interface : Interface) (offset source : Nat) (inputs : Assumptions interface offset) :
    ∀ expression ∈ flatConstraints
      (Circuit.ops (Scalar.circuit (childInterface interface offset source) source).main (sourceOffset offset source)),
      expression.VarsBelow (sourceOffset offset source +
        localLength (Circuit.ops (Scalar.circuit (childInterface interface offset source) source).main
          (sourceOffset offset source))) := by
  change ∀ expression ∈ flatConstraints (Scalar.operations _ _ _),
    expression.VarsBelow (_ + localLength (Scalar.operations _ _ _))
  rw [Scalar.localLength_eq]
  exact Scalar.scope _ _ _ (child_inputs interface offset source inputs)

theorem complete_prefix (interface : Interface) (env : Env) (offset : Nat)
    (inputs : Assumptions interface offset) (count : Nat) :
    ∃ completed : Sequence.Prefix env offset, completed.operations = prefixOps interface offset count := by
  induction count with
  | zero => exact ⟨Sequence.empty env offset, rfl⟩
  | succ count ih =>
      obtain ⟨before, beforeOps⟩ := ih
      obtain ⟨built, agreement, rows⟩ := Scalar.complete (childInterface interface offset count) count
        before.current (sourceOffset offset count) (child_inputs interface offset count inputs)
      obtain ⟨after, afterOps, _, _, _⟩ := Sequence.appendBuiltAt before (childName count)
        (Scalar.circuit (childInterface interface offset count) count) (sourceOffset offset count)
        (by rw [beforeOps, prefix_length]; rfl)
        (child_scope interface offset count inputs) built agreement rows
      refine ⟨after, ?_⟩
      rw [afterOps, beforeOps, prefix_succ]
      rfl

theorem complete (interface : Interface) (env : Env) (offset : Nat) (inputs : Assumptions interface offset) :
    ∃ completed, AgreesOutside env completed offset (localLength (operations interface offset)) ∧
      holdsFlat completed (operations interface offset) := by
  obtain ⟨done, same⟩ := complete_prefix interface env offset inputs sourceCount
  exact ⟨done.current, by simpa only [operations, ← same] using done.agrees,
    by simpa only [operations, ← same] using done.rows⟩

theorem state_sound (interface : Interface) (env : Env) (offset count : Nat)
    (inputs : Assumptions interface offset) (rows : holds env (prefixOps interface offset count)) :
    ∀ position, position ≤ count →
      Scalar.evalState env (stateAtExpr interface offset position) =
        Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.Transcript.stateAt
          (Scalar.evalState env (interface.initialState offset)) position := by
  intro position bound
  induction position with
  | zero => rfl
  | succ position ih =>
      have child := rows (childOp interface offset position)
        (child_member interface offset count position (by omega))
      have spec : Scalar.SpecHolds (childInterface interface offset position) position
          (sourceOffset offset position) env := child (child_inputs interface offset position inputs)
      have same := spec.state
      change Scalar.evalState env (stateAtExpr interface offset (position + 1)) =
        Scalar.referenceNext (Scalar.evalState env (stateAtExpr interface offset position)) position at same
      rw [ih (by omega)] at same
      exact same

theorem soundness (interface : Interface) (env : Env) (offset : Nat)
    (inputs : Assumptions interface offset) (rows : holds env (operations interface offset)) :
    SpecHolds interface offset env := by
  have states := state_sound interface env offset sourceCount inputs rows
  refine ⟨states sourceCount (by rfl), ?_⟩
  intro source position
  have child := rows (childOp interface offset source.val)
    (child_member interface offset sourceCount source.val source.isLt)
  have spec : Scalar.SpecHolds (childInterface interface offset source.val) source.val
      (sourceOffset offset source.val) env := child (child_inputs interface offset source.val inputs)
  have digit := spec.digits position
  change WideReduction.digitValue env _ _ =
    (Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.sample
      (Scalar.referenceBlock (Scalar.referenceEnter
        (Scalar.evalState env (stateAtExpr interface offset source.val)) source.val)) position).val at digit
  rw [states source.val source.isLt.le] at digit
  exact digit

theorem centered_digit (coefficient :
    Spec.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.Coefficient) :
    WideReduction.fieldOfNat coefficient.val - (2 : F) = Phi81StrongSet.embedCoefficient coefficient := by
  fin_cases coefficient <;> decide

theorem outputChallenge_eval (interface : Interface) (env : Env) (offset : Nat)
    (spec : SpecHolds interface offset env) (source : Fin sourceCount) :
    (fun position => (outputChallenge offset source position).eval env) =
      Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.Transcript.challengeAt
        (Scalar.evalState env (interface.initialState offset)) source.val := by
  funext position
  change (WideReduction.Program.outputWord _ position - 2).eval env = _
  rw [Expr.eval_sub, WideReduction.Program.outputWord_eval, spec.digits source position]
  exact centered_digit _

theorem outputChallenge_member (interface : Interface) (env : Env) (offset : Nat)
    (spec : SpecHolds interface offset env) (source : Fin sourceCount) :
    Phi81StrongSet.ProductionMember (fun position => (outputChallenge offset source position).eval env) := by
  rw [outputChallenge_eval interface env offset spec source]
  exact Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.Transcript.challengeAt_member _ _

theorem outputChallenge_below (offset : Nat) (source : Fin sourceCount) (position : Fin ringDegree) :
    (outputChallenge offset source position).VarsBelow (offset + privateCount) := by
  apply Expr.VarsBelow.mono _
    (WideReduction.Program.outputChallenge_varsBelow (Scalar.rangeOffset (sourceOffset offset source.val)) position)
  have bound := Nat.mul_le_mul_right Scalar.privateCount source.isLt
  rw [Nat.succ_mul] at bound
  change offset + source.val * Scalar.privateCount + 592 + WideReduction.Program.privateCount ≤
    offset + sourceCount * Scalar.privateCount
  have localBound : 592 + WideReduction.Program.privateCount ≤ Scalar.privateCount := by
    unfold Scalar.privateCount
    omega
  omega

theorem scope (interface : Interface) (offset : Nat) (inputs : Assumptions interface offset) :
    ∀ expression ∈ flatConstraints (operations interface offset), expression.VarsBelow (offset + privateCount) := by
  intro expression member
  obtain ⟨operation, inside, member⟩ := List.mem_flatMap.mp member
  obtain ⟨source, sourceMember, rfl⟩ := List.mem_map.mp inside
  have sourceBound := List.mem_range.mp sourceMember
  have bounded := Scalar.scope (childInterface interface offset source) source (sourceOffset offset source)
    (child_inputs interface offset source inputs) expression member
  apply Expr.VarsBelow.mono _ bounded
  unfold sourceOffset privateCount
  have countBound : (source + 1) * Scalar.privateCount ≤ sourceCount * Scalar.privateCount :=
    Nat.mul_le_mul_right _ sourceBound
  rw [Nat.add_mul, Nat.one_mul] at countBound
  omega

def circuit (interface : Interface) : FormalCircuit where
  main := fun offset => ((), offset + privateCount, operations interface offset)
  assumptions := fun offset _ => Assumptions interface offset
  spec := SpecHolds interface
  privateCount := fun _ => privateCount
  rowCount := fun _ => logicalRowCount
  privateCount_eq := localLength_eq interface
  rowCount_eq := rowCount_eq interface
  soundness := soundness interface
  completeness := fun env offset inputs _ => complete interface env offset inputs

theorem circuit_ops (interface : Interface) (offset : Nat) :
    Circuit.ops (circuit interface).main offset = operations interface offset := rfl

end NightstreamFPrime.Lifecycle.PiRLC.Wide.Batch

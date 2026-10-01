import NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler
import NightstreamFPrime.Lifecycle.Types

/-! The 17 transcript-chained PiRLC sampler calls. The outgoing state of
each call is the next call's input; centered coefficients are expression
views of the checked digits. No copy or boundary rows are added. -/

namespace NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerChain

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Sampling

abbrev Interface := Sampler.Interface
abbrev sourceCount := productionShape.sourceCount
abbrev EState := Sampler.EState

def logicalPrivateCount : Nat := sourceCount * Sampler.logicalPrivateCount
def logicalRowCount : Nat := sourceCount * Sampler.logicalRowCount
def sourceOffset (offset source : Nat) : Nat := offset + source * Sampler.logicalPrivateCount

def stateAtExpr (interface : Interface) (offset : Nat) : Nat → EState
  | 0 => interface.initialState offset
  | source + 1 => Sampler.outputState
      { initialState := fun _ => stateAtExpr interface offset source }
      source (sourceOffset offset source)

def evalInitialState (interface : Interface) (offset : Nat) (env : Env) : Poseidon2.State :=
  Sampler.evalState env (interface.initialState offset)

def childInterface (interface : Interface) (offset source : Nat) : Sampler.Interface where
  initialState := fun _ => stateAtExpr interface offset source

def childName (source : Nat) : String := "pirlc.v1_1.sampler.scalar_" ++ toString source

def childOp (interface : Interface) (offset source : Nat) : Op :=
  Sequence.childOp (childName source) (Sampler.circuit (childInterface interface offset source) source)
    (sourceOffset offset source)

def prefixOps (interface : Interface) (offset count : Nat) : List Op :=
  (List.range count).map (childOp interface offset)

def opsAt (interface : Interface) (offset : Nat) : List Op := prefixOps interface offset sourceCount

def outputChallenge (offset : Nat) (source : Fin sourceCount) : Fin ringDegree → Expr :=
  Sampler.outputChallenge (sourceOffset offset source.val)

def outputState (interface : Interface) (offset : Nat) : EState := stateAtExpr interface offset sourceCount

abbrev Assumptions := Sampler.Assumptions

structure SpecHolds (interface : Interface) (offset : Nat) (env : Env) : Prop where
  state : Sampler.evalState env (outputState interface offset) =
    Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.stateAt
      (Sampler.evalState env (interface.initialState offset)) sourceCount
  digits : ∀ source : Fin sourceCount, ∀ position : Fin ringDegree,
    (Sampler.outputWord (sourceOffset offset source.val) position).eval env =
      WideReduction.fieldOfNat ((Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.scalarAt
        (Sampler.evalState env (interface.initialState offset)) source.val position).val)

theorem sourceCount_eq : sourceCount = 17 := by
  rw [sourceCount, productionShape_sourceCount]
  rfl

theorem counts : logicalPrivateCount = 72539 ∧ logicalRowCount = 49759 := by
  rw [logicalPrivateCount, logicalRowCount, sourceCount_eq, Sampler.counts.1, Sampler.counts.2]
  decide

private theorem child_length (interface : Interface) (offset source : Nat) :
    (childOp interface offset source).localLength = Sampler.logicalPrivateCount := by
  rw [childOp, Sequence.childOp_localLength]
  exact Sampler.localLength_eq _ _ _

private theorem child_rows (interface : Interface) (offset source : Nat) :
    (childOp interface offset source).rowCount = Sampler.logicalRowCount := rfl

theorem prefix_length (interface : Interface) (offset count : Nat) :
    localLength (prefixOps interface offset count) = count * Sampler.logicalPrivateCount := by
  simp only [prefixOps, localLength, List.map_map, Function.comp_def, child_length,
    List.map_const', List.length_range, List.sum_replicate, smul_eq_mul]

theorem localLength_eq (interface : Interface) (offset : Nat) :
    localLength (opsAt interface offset) = logicalPrivateCount := prefix_length _ _ _

theorem rowCount_eq (interface : Interface) (offset : Nat) :
    (flatConstraints (opsAt interface offset)).length = logicalRowCount := by
  rw [flatConstraints_length_eq_rowCount]
  simp only [opsAt, prefixOps, Circuit.rowCount, List.map_map, Function.comp_def, child_rows,
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
      have below := Sampler.outputState_below (childInterface interface offset count) count
        (sourceOffset offset count) ih
      simpa only [stateAtExpr, sourceOffset, Nat.add_mul, Nat.one_mul, Nat.add_assoc] using! below

theorem child_inputs (interface : Interface) (offset source : Nat) (inputs : Assumptions interface offset) :
    Sampler.Assumptions (childInterface interface offset source) (sourceOffset offset source) :=
  state_below interface offset inputs source

theorem child_scope (interface : Interface) (offset source : Nat) (inputs : Assumptions interface offset) :
    ∀ expression ∈ flatConstraints
      (Circuit.ops (Sampler.circuit (childInterface interface offset source) source).main (sourceOffset offset source)),
      expression.VarsBelow (sourceOffset offset source +
        localLength (Circuit.ops (Sampler.circuit (childInterface interface offset source) source).main
          (sourceOffset offset source))) := by
  change ∀ expression ∈ flatConstraints (Sampler.opsAt _ _ _),
    expression.VarsBelow (_ + localLength (Sampler.opsAt _ _ _))
  rw [Sampler.localLength_eq]
  exact Sampler.scope _ _ _ (child_inputs interface offset source inputs)

theorem complete_prefix (interface : Interface) (env : Env) (offset : Nat)
    (inputs : Assumptions interface offset) (count : Nat) :
    ∃ completed : Sequence.Prefix env offset, completed.operations = prefixOps interface offset count := by
  induction count with
  | zero => exact ⟨Sequence.empty env offset, rfl⟩
  | succ count ih =>
      obtain ⟨before, beforeOps⟩ := ih
      obtain ⟨built, agreement, rows⟩ := Sampler.complete (childInterface interface offset count) count
        before.current (sourceOffset offset count) (child_inputs interface offset count inputs)
      obtain ⟨after, afterOps, _, _, _⟩ := Sequence.appendBuiltAt before (childName count)
        (Sampler.circuit (childInterface interface offset count) count) (sourceOffset offset count)
        (by rw [beforeOps, prefix_length]; rfl)
        (child_scope interface offset count inputs) built agreement rows
      refine ⟨after, ?_⟩
      rw [afterOps, beforeOps, prefix_succ]
      rfl

theorem complete (interface : Interface) (env : Env) (offset : Nat) (inputs : Assumptions interface offset) :
    ∃ completed, AgreesOutside env completed offset (localLength (opsAt interface offset)) ∧
      holdsFlat completed (opsAt interface offset) := by
  obtain ⟨done, same⟩ := complete_prefix interface env offset inputs sourceCount
  exact ⟨done.current, by simpa only [opsAt, ← same] using done.agrees,
    by simpa only [opsAt, ← same] using done.rows⟩

theorem state_sound (interface : Interface) (env : Env) (offset count : Nat)
    (inputs : Assumptions interface offset) (rows : holds env (prefixOps interface offset count)) :
    ∀ position, position ≤ count →
      Sampler.evalState env (stateAtExpr interface offset position) =
        Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.stateAt
          (Sampler.evalState env (interface.initialState offset)) position := by
  intro position bound
  induction position with
  | zero => rfl
  | succ position ih =>
      have child := rows (childOp interface offset position)
        (child_member interface offset count position (by omega))
      have spec : Sampler.SpecHolds (childInterface interface offset position) position
          (sourceOffset offset position) env := child (child_inputs interface offset position inputs)
      have same := spec.state
      change Sampler.evalState env (stateAtExpr interface offset (position + 1)) =
        Sampler.referenceNext (Sampler.evalState env (stateAtExpr interface offset position)) position at same
      rw [ih (by omega)] at same
      exact same

theorem soundness (interface : Interface) (env : Env) (offset : Nat)
    (inputs : Assumptions interface offset) (rows : holds env (opsAt interface offset)) :
    SpecHolds interface offset env := by
  have states := state_sound interface env offset sourceCount inputs rows
  refine ⟨states sourceCount (by rfl), ?_⟩
  intro source position
  have child := rows (childOp interface offset source.val)
    (child_member interface offset sourceCount source.val source.isLt)
  have spec : Sampler.SpecHolds (childInterface interface offset source.val) source.val
      (sourceOffset offset source.val) env := child (child_inputs interface offset source.val inputs)
  have digit := spec.digits position
  change (Sampler.outputWord _ _).eval env =
    WideReduction.fieldOfNat ((Spec.Folding.Nifs.NonInteractive.PiRlcSampler.sample
      (Sampler.referenceBlock (Sampler.referenceEnter
        (Sampler.evalState env (stateAtExpr interface offset source.val)) source.val)) position).val) at digit
  rw [states source.val source.isLt.le] at digit
  exact digit

/-- Composition uses the same scalar relation as the logical circuit. -/
theorem spec_of_children (interface : Interface) (env : Env) (offset : Nat)
    (inputs : Assumptions interface offset)
    (children : ∀ source : Fin sourceCount,
      Sampler.SpecHolds (childInterface interface offset source.val) source.val
        (sourceOffset offset source.val) env) : SpecHolds interface offset env := by
  apply soundness interface env offset inputs
  intro operation member
  simp only [opsAt, prefixOps, List.mem_map, List.mem_range] at member
  obtain ⟨source, sourceLt, rfl⟩ := member
  intro _
  exact children ⟨source, sourceLt⟩

theorem centered_digit (coefficient :
    Phi81StrongSet.Coefficient) :
    WideReduction.fieldOfNat coefficient.val - (2 : F) = Phi81StrongSet.embedCoefficient coefficient := by
  fin_cases coefficient <;> decide

theorem outputChallenge_eval (interface : Interface) (env : Env) (offset : Nat)
    (spec : SpecHolds interface offset env) (source : Fin sourceCount) :
    (fun position => (outputChallenge offset source position).eval env) =
      Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.challengeAt
        (Sampler.evalState env (interface.initialState offset)) source.val := by
  funext position
  change (Sampler.outputWord _ position - 2).eval env = _
  rw [Expr.eval_sub, spec.digits source position]
  exact centered_digit _

theorem outputChallenge_member (interface : Interface) (env : Env) (offset : Nat)
    (spec : SpecHolds interface offset env) (source : Fin sourceCount) :
    Phi81StrongSet.ProductionMember (fun position => (outputChallenge offset source position).eval env) := by
  rw [outputChallenge_eval interface env offset spec source]
  exact Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.challengeAt_member _ _

theorem outputChallenge_below (offset : Nat) (source : Fin sourceCount) (position : Fin ringDegree) :
    (outputChallenge offset source position).VarsBelow (offset + logicalPrivateCount) := by
  apply Expr.VarsBelow.mono _ (Sampler.outputChallenge_below (sourceOffset offset source.val) position)
  have bound := Nat.mul_le_mul_right Sampler.logicalPrivateCount source.isLt
  rw [Nat.succ_mul] at bound
  unfold sourceOffset logicalPrivateCount
  omega

theorem scope (interface : Interface) (offset : Nat) (inputs : Assumptions interface offset) :
    ∀ expression ∈ flatConstraints (opsAt interface offset), expression.VarsBelow (offset + logicalPrivateCount) := by
  intro expression member
  obtain ⟨operation, inside, member⟩ := List.mem_flatMap.mp member
  obtain ⟨source, sourceMember, rfl⟩ := List.mem_map.mp inside
  have sourceBound := List.mem_range.mp sourceMember
  have bounded := Sampler.scope (childInterface interface offset source) source (sourceOffset offset source)
    (child_inputs interface offset source inputs) expression member
  apply Expr.VarsBelow.mono _ bounded
  unfold sourceOffset logicalPrivateCount
  have countBound : (source + 1) * Sampler.logicalPrivateCount ≤ sourceCount * Sampler.logicalPrivateCount :=
    Nat.mul_le_mul_right _ sourceBound
  rw [Nat.add_mul, Nat.one_mul] at countBound
  omega

def main (interface : Interface) : Circuit Unit :=
  fun offset => ((), offset + logicalPrivateCount, opsAt interface offset)

def circuit (interface : Interface) : FormalCircuit where
  main := main interface
  assumptions := fun offset _ => Assumptions interface offset
  spec := SpecHolds interface
  privateCount := fun _ => logicalPrivateCount
  rowCount := fun _ => logicalRowCount
  privateCount_eq := localLength_eq interface
  rowCount_eq := rowCount_eq interface
  soundness := soundness interface
  completeness := fun env offset inputs _ => complete interface env offset inputs

theorem circuit_ops (interface : Interface) (offset : Nat) :
    Circuit.ops (circuit interface).main offset = opsAt interface offset := rfl

end NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerChain

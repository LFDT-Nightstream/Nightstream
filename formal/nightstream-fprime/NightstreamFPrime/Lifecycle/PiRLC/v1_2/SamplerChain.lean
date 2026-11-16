import NightstreamFPrime.Lifecycle.PiRLC.v1_2.Sampler
import NightstreamFPrime.Lifecycle.Types

namespace NightstreamFPrime.Lifecycle.PiRLC.v1_2.SamplerChain

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

def childName (source : Nat) : String := "pirlc.v1_2.sampler.scalar_" ++ toString source

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

theorem sourceCount_eq : sourceCount = 15 := by
  rw [sourceCount, productionShape_sourceCount]
  rfl

theorem counts : logicalPrivateCount = 73095 ∧ logicalRowCount = 46515 := by
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
    RingStrongSet.Coefficient) :
    WideReduction.fieldOfNat coefficient.val - (2 : F) = RingStrongSet.embedCoefficient coefficient := by
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
    RingStrongSet.ProductionMember (fun position => (outputChallenge offset source position).eval env) := by
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

/-! ### The exporter's blocks (SPEC.md section 4.4) -/

open NightstreamFPrime.Gadgets.Poseidon2

theorem child_localLength (interface : Interface) (offset count : Nat) :
    localLength (Circuit.ops (Sampler.circuit (childInterface interface offset count) count).main
      (sourceOffset offset count)) = Sampler.logicalPrivateCount :=
  Sampler.localLength_eq _ _ _

theorem child_placed (interface : Interface) (offset count : Nat) :
    Placed (offset + count * Sampler.logicalPrivateCount)
      (localLength [childOp interface offset count]) [childOp interface offset count] :=
  Placed.child_cons (child_localLength interface offset count) (Sampler.placed _ _ _)
    (Placed.nil _)

theorem prefix_placed (interface : Interface) (offset count : Nat) :
    Placed offset (localLength (prefixOps interface offset count))
      (prefixOps interface offset count) := by
  induction count with
  | zero => exact Placed.nil offset
  | succ count ih =>
      rw [prefix_succ, Sequence.localLength_append]
      refine ih.append ?_
      rw [prefix_length]
      exact child_placed interface offset count

/-- The first `count` samplers' rows, their challenge words and the state after them. -/
theorem prefix_liveRows (interface : Interface) (offset : Nat) {live : Nat → Prop}
    (stateFacts : Duplex.Formal.StateFacts live offset (interface.initialState offset))
    (count : Nat)
    (own : OwnLive live offset (localLength (prefixOps interface offset count))
      (prefixOps interface offset count)) :
    LiveRows live (prefixOps interface offset count) ∧
      (∀ source, source < count → ∀ position,
        (Sampler.outputChallenge (sourceOffset offset source) position).VarsSatisfy live) ∧
      Duplex.Formal.StateFacts live (sourceOffset offset count)
        (stateAtExpr interface offset count) := by
  induction count with
  | zero =>
      exact ⟨LiveRows.nil, fun _ below => absurd below (Nat.not_lt_zero _),
        stateFacts.mono (Nat.le_add_right _ _)⟩
  | succ count ih =>
      rw [prefix_succ, Sequence.localLength_append] at own
      have childPlaced := child_placed interface offset count
      rw [← prefix_length] at childPlaced
      obtain ⟨prefixLiveRows, challenges, state⟩ := ih (own.left childPlaced)
      have childOwn := own.right (prefix_placed interface offset count)
      rw [prefix_length] at childOwn
      obtain ⟨samplerOwn, -⟩ := childOwn.child_cons (child_localLength interface offset count)
        (Sampler.placed _ _ _) (Placed.nil _)
      obtain ⟨samplerLiveRows, challenge, next⟩ := Sampler.liveRows
        (childInterface interface offset count) count (sourceOffset offset count) state samplerOwn
      rw [prefix_succ]
      refine ⟨prefixLiveRows.append (LiveRows.child_cons samplerLiveRows LiveRows.nil),
        fun source below => ?_, next.mono (Nat.le_of_eq ?_)⟩
      · rcases Nat.lt_succ_iff_lt_or_eq.mp below with below | rfl
        · exact challenges source below
        · exact challenge
      · simp only [sourceOffset, Nat.add_mul, Nat.one_mul, Nat.add_assoc]

theorem circuit_localLength (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (circuit interface).main offset) = logicalPrivateCount :=
  localLength_eq interface offset

theorem placed (interface : Interface) (offset : Nat) :
    Placed offset (localLength (Circuit.ops (circuit interface).main offset))
      (Circuit.ops (circuit interface).main offset) :=
  prefix_placed interface offset sourceCount

/-- The chain's rows, every sampler's challenge words and the state after the chain. -/
theorem liveRows (interface : Interface) (offset : Nat) {live : Nat → Prop}
    (stateFacts : Duplex.Formal.StateFacts live offset (interface.initialState offset))
    (own : OwnLive live offset (localLength (Circuit.ops (circuit interface).main offset))
      (Circuit.ops (circuit interface).main offset)) :
    LiveRows live (Circuit.ops (circuit interface).main offset) ∧
      (∀ source position, (outputChallenge offset source position).VarsSatisfy live) ∧
      Duplex.Formal.StateFacts live (offset + logicalPrivateCount)
        (outputState interface offset) := by
  obtain ⟨prefixLiveRows, challenges, state⟩ :=
    prefix_liveRows interface offset stateFacts sourceCount own
  exact ⟨prefixLiveRows, fun source position => challenges source.val source.isLt position, state⟩

end NightstreamFPrime.Lifecycle.PiRLC.v1_2.SamplerChain

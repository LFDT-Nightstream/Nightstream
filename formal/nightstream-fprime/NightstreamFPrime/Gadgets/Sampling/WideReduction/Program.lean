import NightstreamFPrime.Gadgets.Sampling.WideReduction.OutputExecution
import NightstreamFPrime.Gadgets.Sampling.WideReduction.CanonicalChildren
import NightstreamFPrime.Gadgets.Sampling.WideReduction.ReadSupport
import NightstreamFPrime.Gadgets.Sampling.WideReduction.HelperExecution

/-! Executable wide-sampler circuit. Temporary hints precede one opaque
checked gadget. Production selection and retained-column placement are
separate obligations. -/

namespace NightstreamFPrime.Gadgets.Sampling.WideReduction.Program

open NightstreamFPrime.Spec NightstreamFPrime.Circuit

def coreOffset (start : Nat) : Nat := start + HintProgram.helperCount

def coreInterface (interface : Interface) (start : Nat) : Interface where
  source := fun lane _ => interface.source lane start

def checked (interface : Interface) (start : Nat) : FormalCircuit :=
  WideReduction.circuit (coreInterface interface start) (HintProgram.resultHints start)
    (HintProgram.resultHints_length start)

def checkedOp (interface : Interface) (start : Nat) : Op :=
  Sequence.childOp "pirlc.wide_sampler.checked" (checked interface start) (coreOffset start)

def operations (interface : Interface) (start : Nat) : List Op :=
  [.witness (WitnessBatch.hinted start (HintProgram.helpers interface start)), checkedOp interface start]

def privateCount : Nat := HintProgram.helperCount + WideReduction.privateCount

theorem privateCount_eq : privateCount = 2617 := rfl

theorem localLength_eq (interface : Interface) (start : Nat) :
    localLength (operations interface start) = privateCount := by
  simp only [operations, localLength, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
    Op.localLength, WitnessBatch.hinted, WitnessBatch.outputLength, List.length_nil,
    HintProgram.helpers_length, Nat.zero_add, Nat.add_zero]
  rfl

theorem constraints_eq (interface : Interface) (start : Nat) :
    flatConstraints (operations interface start) =
      flatConstraints (WideReduction.operations (coreInterface interface start)
        (HintProgram.resultHints start) (coreOffset start)) := by
  change [] ++ ((checkedOp interface start).flatConstraints ++ []) = _
  rw [List.nil_append, List.append_nil]
  rfl

theorem flatConstraints_varsSatisfy (interface : Interface) (start : Nat) (allowed : Nat → Prop)
    (sources : ∀ lane, (interface.source lane start).VarsSatisfy allowed)
    (locals : ∀ index, coreOffset start ≤ index →
      index < coreOffset start + WideReduction.privateCount → allowed index) :
    ∀ expression ∈ flatConstraints (operations interface start), expression.VarsSatisfy allowed := by
  rw [constraints_eq]
  exact WideReduction.flatConstraints_varsSatisfy _ _ _ allowed sources locals

theorem rowCount_eq (interface : Interface) (start : Nat) :
    (flatConstraints (operations interface start)).length = WideReduction.rowCount := by
  rw [constraints_eq, WideReduction.rowCount_eq]

theorem temporary_read_exclusion (interface : Interface) (start : Nat)
    (inputs : Assumptions interface start) (env : Env) (temporary : Nat → F) :
    holdsFlat (replaceTemporary env start HintProgram.helperCount temporary) (operations interface start) ↔
      holdsFlat env (operations interface start) := by
  change ConstraintsHold _ (flatConstraints _) ↔ ConstraintsHold _ (flatConstraints _)
  rw [constraints_eq]
  exact WideReduction.temporary_read_exclusion (coreInterface interface start)
    (HintProgram.resultHints start) start HintProgram.helperCount inputs env temporary

theorem helper_reconstruction (interface : Interface) (start : Nat)
    (inputs : Assumptions interface start) (env : Env) :
    holdsFlat (HintProgram.helperEnv interface env start) (operations interface start) ↔
      holdsFlat env (operations interface start) := by
  have replacement : HintProgram.helperEnv interface env start =
      replaceTemporary env start HintProgram.helperCount
        (fun index => HintProgram.helperEnv interface env start (start + index)) := by
    funext index
    unfold replaceTemporary
    split_ifs with inside
    · dsimp only
      rw [Nat.add_sub_of_le inside.1]
    · exact HintProgram.helperEnv_agreesOutside interface env start index (by omega)
  rw [replacement]
  exact temporary_read_exclusion interface start inputs env _

theorem core_inputs (interface : Interface) (start : Nat) (inputs : Assumptions interface start) :
    Assumptions (coreInterface interface start) (coreOffset start) := by
  intro lane
  exact Expr.VarsBelow.mono _ (inputs lane) (by unfold coreOffset; omega)

theorem flatConstraints_varsBelow (interface : Interface) (start : Nat)
    (inputs : Assumptions interface start) :
    ∀ expression ∈ flatConstraints (operations interface start),
      expression.VarsBelow (start + privateCount) := by
  rw [constraints_eq]
  have scope := WideReduction.flatConstraints_varsBelow (coreInterface interface start)
    (HintProgram.resultHints start) (coreOffset start) (core_inputs interface start inputs)
  simpa only [coreOffset, privateCount, Nat.add_assoc] using scope

def outputWord (start : Nat) (position : Fin ringDegree) : Expr :=
  linearExpr [(1, digitBit (coreOffset start) position.val 0),
    (2, digitBit (coreOffset start) position.val 1),
    (4, digitBit (coreOffset start) position.val 2)]

def outputChallenge (start : Nat) (position : Fin ringDegree) : Expr :=
  outputWord start position - 2

theorem outputWord_varsSatisfy (start : Nat) (position : Fin ringDegree) (allowed : Nat → Prop)
    (locals : ∀ index, coreOffset start ≤ index →
      index < coreOffset start + WideReduction.privateCount → allowed index) :
    (outputWord start position).VarsSatisfy allowed := by
  have bits (bit : Nat) (bound : bit < 3) :
      allowed (digitStart (coreOffset start) + digitBitCount * position.val + bit) := by
    have positionLt := position.isLt
    apply locals <;>
      norm_num [digitStart, quotientStart, childWidth, Range.CanonicalU64.auxiliaryCount,
        fieldCount, quotientBitCount, digitBitCount, WideReduction.privateCount,
        newBitCount, digitCount, checkCount, checkBitCount, ringDegree] at * <;> omega
  unfold outputWord linearExpr
  exact ⟨⟨trivial, bits 0 (by decide)⟩,
    ⟨⟨trivial, bits 1 (by decide)⟩, ⟨⟨trivial, bits 2 (by decide)⟩, trivial⟩⟩⟩

theorem outputWord_eval (env : Env) (start : Nat) (position : Fin ringDegree) :
    (outputWord start position).eval env = fieldOfNat (digitValue env (coreOffset start) position.val) := by
  rw [outputWord, linearExpr_eval]
  apply congrArg fieldOfNat
  simp only [linearValue, digitValue, Nat.one_mul, Nat.add_zero]
  omega

theorem outputChallenge_varsBelow (start : Nat) (position : Fin ringDegree) :
    (outputChallenge start position).VarsBelow (start + privateCount) := by
  have bits (bit : Nat) (bound : bit < digitBitCount) :
      (digitBit (coreOffset start) position.val bit).VarsBelow (start + privateCount) := by
    change _ < _
    have positionBound := position.isLt
    simp only [digitStart, quotientStart, coreOffset, privateCount, WideReduction.privateCount,
      newBitCount, quotientBitCount, digitCount, digitBitCount, checkCount, checkBitCount,
      ringDegree] at *
    omega
  refine Expr.VarsBelow.sub _ _ _ ?_ trivial
  apply HintSupport.linearExpr_below
  intro term member
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl
  · exact bits 0 (by decide)
  · exact bits 1 (by decide)
  · exact bits 2 (by decide)

theorem witnesses_operations (interface : Interface) (start : Nat) :
    witnesses (operations interface start) =
      WitnessBatch.hinted start (HintProgram.helpers interface start) ::
        (witnesses (childOps (coreInterface interface start) (coreOffset start)) ++
          [WitnessBatch.hinted (quotientStart (coreOffset start))
            (HintProgram.resultHints start (coreOffset start))]) := by
  have rows : witnesses (WideReduction.rowOps (coreOffset start)) = [] := by
    unfold witnesses WideReduction.rowOps
    rw [List.flatMap_eq_nil_iff]
    intro op member
    simp only [List.mem_append, List.mem_map] at member
    rcases member with (⟨_, _, rfl⟩ | ⟨_, _, rfl⟩) | ⟨_, _, rfl⟩ <;> rfl
  change _ :: (witnesses (WideReduction.operations (coreInterface interface start)
    (HintProgram.resultHints start) (coreOffset start)) ++ []) = _
  rw [List.append_nil, WideReduction.operations, Witness.witnesses_append,
    Witness.witnesses_append, rows, List.append_nil]
  rfl

theorem ordered (interface : Interface) (start : Nat) (inputs : Assumptions interface start) :
    Witness.Ordered start (witnesses (operations interface start))
      (start + localLength (operations interface start)) := by
  rw [witnesses_operations, localLength_eq]
  exact ⟨le_rfl, ⟨trivial, (HelperExecution.certificate interface (fun _ => 0) start inputs).1⟩,
    ((CanonicalChildren.ordered _ _ (core_inputs interface start inputs)).mono
      (by simp [WitnessBatch.finish, WitnessBatch.hinted, WitnessBatch.outputLength,
        HintProgram.helpers_length, coreOffset]) le_rfl).append
      ⟨le_rfl, ⟨trivial, OutputExecution.causal start (coreOffset start) le_rfl⟩, by
        simp [Witness.Ordered, WitnessBatch.finish, WitnessBatch.hinted,
          WitnessBatch.outputLength, HintProgram.resultHints_length, privateCount,
          WideReduction.privateCount, quotientStart, coreOffset, Nat.add_assoc]⟩⟩

theorem computed_holdsFlat (interface : Interface) (env : Env) (start : Nat)
    (inputs : Assumptions interface start)
    (computed : Witness.Computed env (witnesses (operations interface start))) :
    holdsFlat env (operations interface start) := by
  rw [witnesses_operations] at computed
  have later : Witness.Computed env _ := fun batch member =>
    computed batch (List.mem_cons_of_mem _ member)
  rw [Witness.computed_append] at later
  have coreInputs := core_inputs interface start inputs
  have childRows := CanonicalChildren.rows (coreInterface interface start) env (coreOffset start)
    later.1
  have childSpecs := CanonicalChildren.specifications _ env _ coreInputs childRows
  have childScope := CanonicalChildren.scope _ _ coreInputs
  have rows := completeNew_holds (coreInterface interface start) (HintProgram.resultHints start)
    env (coreOffset start) childSpecs childRows childScope
  rw [OutputExecution.reference (coreInterface interface start) env start (coreOffset start)
    le_rfl (HelperExecution.present interface env start inputs
      (computed _ List.mem_cons_self).2) childSpecs childRows childScope
    (later.2 _ (List.mem_singleton.mpr rfl)).2] at rows
  change ConstraintsHold env (flatConstraints (operations interface start))
  rw [constraints_eq]
  exact rows

theorem complete (interface : Interface) (env : Env) (start : Nat)
    (inputs : Assumptions interface start) :
    ∃ completed, AgreesOutside env completed start (localLength (operations interface start)) ∧
      holdsFlat completed (operations interface start) :=
  Witness.complete (ordered interface start inputs) fun completed _ =>
    computed_holdsFlat interface completed start inputs

def circuit (interface : Interface) : FormalCircuit where
  main := fun start => ((), start + privateCount, operations interface start)
  assumptions := fun start _ => Assumptions interface start
  spec := fun start env => SpecHolds (coreInterface interface start) (coreOffset start) env
  privateCount := fun _ => privateCount
  rowCount := fun _ => WideReduction.rowCount
  privateCount_eq := localLength_eq interface
  rowCount_eq := rowCount_eq interface
  soundness := by
    intro env start inputs rows
    change holds env (operations interface start) at rows
    have child := rows (checkedOp interface start) (by simp [operations])
    exact child (core_inputs interface start inputs)
  completeness := fun env start inputs _ => complete interface env start inputs

/-! ### The exporter's blocks (SPEC.md section 4.4) -/

private theorem checked_marks (interface : Interface) (start : Nat) :
    marks (Circuit.ops (checked interface start).main (coreOffset start)) =
      marks (WideReduction.operations (coreInterface interface start)
        (HintProgram.resultHints start) (coreOffset start)) :=
  rfl

theorem marks_nil (interface : Interface) (start : Nat) :
    marks (Circuit.ops (circuit interface).main start) = [] := by
  have children : marks (WideReduction.childOps (coreInterface interface start)
      (coreOffset start)) = [] := by
    unfold marks WideReduction.childOps
    rw [List.flatMap_eq_nil_iff]
    intro op member
    obtain ⟨index, _, rfl⟩ := List.mem_map.mp member
    exact Range.CanonicalU64.marks_nil
      (childInterface (coreInterface interface start) (coreOffset start) index)
      (childOffset (coreOffset start) index)
  have rows : marks (WideReduction.rowOps (coreOffset start)) = [] := by
    unfold marks WideReduction.rowOps
    rw [List.flatMap_eq_nil_iff]
    intro op member
    simp only [List.mem_append, List.mem_map] at member
    rcases member with (⟨_, _, rfl⟩ | ⟨_, _, rfl⟩) | ⟨_, _, rfl⟩ <;> rfl
  have core : marks (WideReduction.operations (coreInterface interface start)
      (HintProgram.resultHints start) (coreOffset start)) = [] := by
    rw [WideReduction.operations, marks_append, marks_append, children, rows]
    rfl
  rw [show Circuit.ops (circuit interface).main start = operations interface start from rfl,
    operations, marks_cons, marks_cons, NightstreamFPrime.Circuit.marks_nil,
    show Op.marks (Op.witness (WitnessBatch.hinted start (HintProgram.helpers interface start))) =
      [] from rfl,
    checkedOp, Sequence.childOp, Op.marks, FormalCircuit.asSubcircuit_marks, checked_marks, core]
  rfl

/-- The rows read the sources and the gadget's own locals. -/
theorem free (interface : Interface) (start : Nat) {live : Nat → Prop}
    (sources : ∀ lane, (interface.source lane start).VarsSatisfy live)
    (locals : ∀ column, start ≤ column →
      column < start + localLength (Circuit.ops (circuit interface).main start) → live column) :
    ∀ row ∈ flatConstraints (Circuit.ops (circuit interface).main start), row.VarsSatisfy live :=
  flatConstraints_varsSatisfy interface start live sources fun column lower upper => by
    have length := localLength_eq interface start
    unfold coreOffset at lower upper
    unfold privateCount at length
    exact locals column (by omega) (by change column < start + localLength (operations _ _); omega)

/-- The output words read only the gadget's own locals. -/
theorem outputWord_live (interface : Interface) (start : Nat) {live : Nat → Prop}
    (locals : ∀ column, start ≤ column →
      column < start + localLength (Circuit.ops (circuit interface).main start) → live column)
    (position : Fin ringDegree) : (outputWord start position).VarsSatisfy live :=
  outputWord_varsSatisfy start position live fun column lower upper => by
    have length := localLength_eq interface start
    unfold coreOffset at lower upper
    unfold privateCount at length
    exact locals column (by omega) (by change column < start + localLength (operations _ _); omega)

end NightstreamFPrime.Gadgets.Sampling.WideReduction.Program

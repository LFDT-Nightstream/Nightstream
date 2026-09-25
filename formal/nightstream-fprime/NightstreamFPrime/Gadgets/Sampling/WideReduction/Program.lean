import NightstreamFPrime.Gadgets.Sampling.WideReduction.OutputExecution
import NightstreamFPrime.Gadgets.Sampling.WideReduction.CanonicalChildren
import NightstreamFPrime.Gadgets.Sampling.WideReduction.ReadSupport

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

theorem privateCount_eq : privateCount = 2021 := rfl

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

def childEnv (interface : Interface) (base : Env) (start : Nat) : Env :=
  CanonicalChildren.values (coreInterface interface start) (HintProgram.helperEnv interface base start)
    (coreOffset start) fieldCount

def completeEnv (interface : Interface) (base : Env) (start : Nat) : Env :=
  executeHints (childEnv interface base start) (quotientStart (coreOffset start))
    (HintProgram.resultHints start (coreOffset start))

theorem completeEnv_correct (interface : Interface) (base : Env) (start : Nat)
    (inputs : Assumptions interface start) :
    AgreesOutside base (completeEnv interface base start) start privateCount ∧
      holdsFlat (completeEnv interface base start) (operations interface start) := by
  let prepared := HintProgram.helperEnv interface base start
  let children := childEnv interface base start
  let core := coreInterface interface start
  have helperAgreement := HintProgram.helperEnv_agreesOutside interface base start
  change AgreesOutside base prepared start HintProgram.helperCount at helperAgreement
  have childAgreement :=
    (CanonicalChildren.correct core prepared (coreOffset start) (core_inputs interface start inputs)
      fieldCount (by rfl)).1
  change AgreesOutside prepared children (coreOffset start) (childWidth * fieldCount) at childAgreement
  have drawSame : drawOf core children (coreOffset start) = drawOf interface base start := by
    funext lane
    apply Expr.eval_eq_of_agree_below _ start children base (inputs lane)
    intro index below
    rw [childAgreement index (Or.inl (by unfold coreOffset; omega)),
      helperAgreement index (Or.inl below)]
  have helperPresent : HelperValues.Present prepared start (drawOf interface base start) := by
    rw [show prepared = HelperValues.environment base start (drawOf interface base start) from
      HelperExecution.execute_reference interface base start inputs]
    exact HelperValues.present base start _
  have childrenPresent : HelperValues.Present children start (drawOf core children (coreOffset start)) := by
    rw [drawSame]
    apply HelperValues.present_of_agree prepared children start _ helperPresent
    intro index _ upper
    exact childAgreement index (Or.inl upper)
  have childRows := CanonicalChildren.rows core prepared (coreOffset start) (core_inputs interface start inputs)
  have childScope := CanonicalChildren.scope core (coreOffset start) (core_inputs interface start inputs)
  have childSpecs := CanonicalChildren.specifications core prepared (coreOffset start)
    (core_inputs interface start inputs)
  have resultSame := OutputExecution.execute_reference core children start (coreOffset start)
    (by rfl) childrenPresent childSpecs childRows childScope
  have resultAgreement := executeHints_agreesOutside children (quotientStart (coreOffset start))
    (HintProgram.resultHints start (coreOffset start))
  rw [HintProgram.resultHints_length] at resultAgreement
  refine ⟨?_, ?_⟩
  · have first := helperAgreement.append childAgreement
    have all := first.append resultAgreement
    simpa only [privateCount, WideReduction.privateCount, quotientStart, coreOffset, Nat.add_assoc] using! all
  · change holdsFlat (executeHints children _ _) _
    rw [resultSame]
    change ConstraintsHold _ (flatConstraints (operations interface start))
    rw [constraints_eq]
    exact completeNew_holds core (HintProgram.resultHints start) children (coreOffset start)
      childSpecs childRows childScope

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
  completeness := by
    intro env start inputs _
    refine ⟨completeEnv interface env start, ?_⟩
    change AgreesOutside env (completeEnv interface env start) start
      (localLength (operations interface start)) ∧
      holdsFlat (completeEnv interface env start) (operations interface start)
    rw [localLength_eq]
    exact completeEnv_correct interface env start inputs

end NightstreamFPrime.Gadgets.Sampling.WideReduction.Program

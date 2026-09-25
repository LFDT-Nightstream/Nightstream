import NightstreamFPrime.Gadgets.Sampling.WideReduction.Program
import NightstreamFPrime.Gadgets.Range.CanonicalU64.Values

/-! Offset-independent readback of the actual range witness program. Equal
four-word inputs produce equal retained canonical and result cells. Temporary
helper cells remain outside this contract. -/

namespace NightstreamFPrime.Gadgets.Sampling.WideReduction.ProgramValues

open NightstreamFPrime.Circuit NightstreamFPrime.Spec
open NightstreamFPrime.Gadgets.Range

private theorem children_source (interface : Interface) (base : Env) (offset count : Nat)
    (inputs : Assumptions interface offset) (bound : count ≤ fieldCount) (lane : Fin fieldCount) :
    (interface.source lane offset).eval (CanonicalChildren.values interface base offset count) =
      (interface.source lane offset).eval base := by
  apply Expr.eval_eq_of_agree_below _ offset _ _ (inputs lane)
  intro index below
  exact (CanonicalChildren.correct interface base offset inputs count bound).1 index (Or.inl below)

private theorem children_local_congr
    (leftInterface rightInterface : Interface) (left right : Env) (leftOffset rightOffset : Nat)
    (leftInputs : Assumptions leftInterface leftOffset)
    (rightInputs : Assumptions rightInterface rightOffset)
    (source : ∀ lane, (leftInterface.source lane leftOffset).eval left =
      (rightInterface.source lane rightOffset).eval right)
    (count : Nat) (bound : count ≤ fieldCount) (lane : Fin fieldCount) (inside : lane.val < count)
    (index : Nat) (bounded : index < CanonicalU64.auxiliaryCount) :
    CanonicalChildren.values leftInterface left leftOffset count (childOffset leftOffset lane.val + index) =
      CanonicalChildren.values rightInterface right rightOffset count (childOffset rightOffset lane.val + index) := by
  induction count with
  | zero => omega
  | succ count ih =>
    have previous : count ≤ fieldCount := by omega
    have current : count < fieldCount := by omega
    have leftAssumptions : CanonicalU64.Assumptions (childInterface leftInterface leftOffset ⟨count, current⟩)
        (childOffset leftOffset count) (CanonicalChildren.values leftInterface left leftOffset count) :=
      Expr.VarsBelow.mono _ (leftInputs ⟨count, current⟩) (by unfold childOffset; omega)
    have rightAssumptions : CanonicalU64.Assumptions (childInterface rightInterface rightOffset ⟨count, current⟩)
        (childOffset rightOffset count) (CanonicalChildren.values rightInterface right rightOffset count) :=
      Expr.VarsBelow.mono _ (rightInputs ⟨count, current⟩) (by unfold childOffset; omega)
    rw [CanonicalChildren.values, dif_pos current, CanonicalChildren.values, dif_pos current]
    by_cases last : lane.val = count
    · have laneEq : lane = ⟨count, current⟩ := Fin.ext last
      subst lane
      apply CanonicalU64.completeEnv_local_congr _ _ _ _ _ _ leftAssumptions rightAssumptions
      · change (leftInterface.source _ leftOffset).eval _ = (rightInterface.source _ rightOffset).eval _
        rw [children_source _ _ _ _ leftInputs previous, children_source _ _ _ _ rightInputs previous]
        exact source _
      · exact bounded
    · have leftBefore : childOffset leftOffset lane.val + index < childOffset leftOffset count := by
        simp only [childOffset, childWidth, CanonicalU64.auxiliaryCount] at *
        omega
      have rightBefore : childOffset rightOffset lane.val + index < childOffset rightOffset count := by
        simp only [childOffset, childWidth, CanonicalU64.auxiliaryCount] at *
        omega
      rw [(CanonicalU64.completeEnv_correct _ _ _ leftAssumptions).1 _ (Or.inl leftBefore),
        (CanonicalU64.completeEnv_correct _ _ _ rightAssumptions).1 _ (Or.inl rightBefore)]
      exact ih previous (by omega)

theorem childEnv_source (interface : Interface) (base : Env) (start : Nat)
    (inputs : Assumptions interface start) (lane : Fin fieldCount) :
    (interface.source lane start).eval (Program.childEnv interface base start) =
      (interface.source lane start).eval base := by
  change ((Program.coreInterface interface start).source lane (Program.coreOffset start)).eval _ = _
  rw [Program.childEnv, children_source _ _ _ _ (Program.core_inputs interface start inputs) (by rfl)]
  apply Expr.eval_eq_of_agree_below _ start _ _ (inputs lane)
  intro index below
  exact HintProgram.helperEnv_agreesOutside interface base start index (Or.inl below)

theorem childEnv_local_congr
    (leftInterface rightInterface : Interface) (left right : Env) (leftStart rightStart : Nat)
    (leftInputs : Assumptions leftInterface leftStart)
    (rightInputs : Assumptions rightInterface rightStart)
    (source : ∀ lane, (leftInterface.source lane leftStart).eval left =
      (rightInterface.source lane rightStart).eval right)
    (lane : Fin fieldCount) (index : Nat) (bounded : index < CanonicalU64.auxiliaryCount) :
    Program.childEnv leftInterface left leftStart (childOffset (Program.coreOffset leftStart) lane.val + index) =
      Program.childEnv rightInterface right rightStart (childOffset (Program.coreOffset rightStart) lane.val + index) := by
  apply children_local_congr _ _ _ _ _ _ (Program.core_inputs _ _ leftInputs)
    (Program.core_inputs _ _ rightInputs) _ fieldCount (by rfl) lane lane.isLt index bounded
  intro lane
  have read (interface : Interface) (base : Env) (start : Nat) (inputs : Assumptions interface start) :
      (interface.source lane start).eval (HintProgram.helperEnv interface base start) =
        (interface.source lane start).eval base := by
    apply Expr.eval_eq_of_agree_below _ start _ _ (inputs lane)
    intro index below
    exact HintProgram.helperEnv_agreesOutside interface base start index (Or.inl below)
  exact (read _ _ _ leftInputs).trans ((source lane).trans (read _ _ _ rightInputs).symm)

/-- The final hint execution is the already proved arithmetic completion. -/
theorem completeEnv_reference (interface : Interface) (base : Env) (start : Nat)
    (inputs : Assumptions interface start) :
    Program.completeEnv interface base start =
      completeNew (Program.coreInterface interface start) (Program.childEnv interface base start)
        (Program.coreOffset start) := by
  let prepared := HintProgram.helperEnv interface base start
  let children := Program.childEnv interface base start
  let core := Program.coreInterface interface start
  have sourceSame : drawOf core children (Program.coreOffset start) = drawOf interface base start := by
    funext lane
    exact childEnv_source interface base start inputs lane
  have preparedHelpers : HelperValues.Present prepared start (drawOf interface base start) := by
    rw [show prepared = HelperValues.environment base start (drawOf interface base start) from
      HelperExecution.execute_reference interface base start inputs]
    exact HelperValues.present base start _
  have helpers : HelperValues.Present children start (drawOf core children (Program.coreOffset start)) := by
    rw [sourceSame]
    apply HelperValues.present_of_agree prepared children start _ preparedHelpers
    intro index _ upper
    exact (CanonicalChildren.correct core prepared (Program.coreOffset start)
      (Program.core_inputs interface start inputs) fieldCount (by rfl)).1 index (Or.inl upper)
  exact OutputExecution.execute_reference core children start (Program.coreOffset start) (by rfl) helpers
    (CanonicalChildren.specifications core prepared (Program.coreOffset start) (Program.core_inputs _ _ inputs))
    (CanonicalChildren.rows core prepared (Program.coreOffset start) (Program.core_inputs _ _ inputs))
    (CanonicalChildren.scope core (Program.coreOffset start) (Program.core_inputs _ _ inputs))

private theorem drawReduction_congr (left right : Env) (leftOffset rightOffset modulus : Nat)
    (cells : ∀ lane : Fin fieldCount, ∀ bit, bit < CanonicalU64.bitCount →
      left (childOffset leftOffset lane.val + bit) = right (childOffset rightOffset lane.val + bit)) :
    linearValue left (reduceTerms modulus (drawTerms leftOffset)) =
      linearValue right (reduceTerms modulus (drawTerms rightOffset)) := by
  have field (lane : Fin fieldCount) :
      linearValue left (reduceTerms modulus (fieldTerms leftOffset lane.val)) =
        linearValue right (reduceTerms modulus (fieldTerms rightOffset lane.val)) := by
    simp only [fieldTerms, reduceTerms, List.map_map, Function.comp_def]
    rw [linearValue_rangeMap, linearValue_rangeMap]
    apply Finset.sum_congr rfl
    intro bit member
    simp only [fieldBit, CanonicalU64.bitExpr, Expr.eval_var]
    rw [cells lane bit (Finset.mem_range.mp member)]
  simpa only [drawTerms, reduceTerms, List.map_append, linearValue_append] using!
    congrArg₂ Nat.add (congrArg₂ Nat.add (congrArg₂ Nat.add (field 0) (field 1)) (field 2)) (field 3)

private theorem resultReduction_congr (left right : Env) (leftOffset rightOffset draw modulus : Nat) :
    linearValue (completedEnv left leftOffset draw (fun _ => 0)) (reduceTerms modulus (resultTerms leftOffset)) =
      linearValue (completedEnv right rightOffset draw (fun _ => 0)) (reduceTerms modulus (resultTerms rightOffset)) := by
  have quotient :
      linearValue (completedEnv left leftOffset draw (fun _ => 0)) (reduceTerms modulus (quotientTerms leftOffset)) =
        linearValue (completedEnv right rightOffset draw (fun _ => 0)) (reduceTerms modulus (quotientTerms rightOffset)) := by
    simp only [quotientTerms, reduceTerms, List.map_map, Function.comp_def]
    rw [linearValue_rangeMap, linearValue_rangeMap]
    apply Finset.sum_congr rfl
    intro bit member
    rw [quotientBit_eval bit (Finset.mem_range.mp member), quotientBit_eval bit (Finset.mem_range.mp member)]
  have digit (position : Nat) (inside : position < digitCount) :
      linearValue (completedEnv left leftOffset draw (fun _ => 0)) (reduceTerms modulus (digitTerms leftOffset position)) =
        linearValue (completedEnv right rightOffset draw (fun _ => 0)) (reduceTerms modulus (digitTerms rightOffset position)) := by
    simp only [digitTerms, reduceTerms, List.map_map, Function.comp_def]
    rw [linearValue_rangeMap, linearValue_rangeMap]
    apply Finset.sum_congr rfl
    intro bit member
    rw [digitBit_eval position bit inside (Finset.mem_range.mp member),
      digitBit_eval position bit inside (Finset.mem_range.mp member)]
  simp only [resultTerms, reduceTerms, List.map_append, List.map_flatMap, linearValue_append]
  apply congrArg₂ Nat.add quotient
  rw [linearValue_flatMap_range, linearValue_flatMap_range]
  exact Finset.sum_congr rfl (fun position member => digit position (Finset.mem_range.mp member))

private theorem checkQuotients_congr (left right : Env) (leftOffset rightOffset draw : Nat)
    (cells : ∀ lane : Fin fieldCount, ∀ bit, bit < CanonicalU64.bitCount →
      left (childOffset leftOffset lane.val + bit) = right (childOffset rightOffset lane.val + bit)) :
    checkQuotients left leftOffset draw = checkQuotients right rightOffset draw := by
  funext index
  unfold checkQuotients
  split
  · dsimp only
    rw [drawReduction_congr left right leftOffset rightOffset _ cells,
      resultReduction_congr left right leftOffset rightOffset draw]
  · rfl

/-- All 617 retained cells agree under an offset change and equal inputs.
The 1,404 preceding helpers are deliberately outside the stated interval. -/
theorem completeEnv_retained_congr
    (leftInterface rightInterface : Interface) (left right : Env) (leftStart rightStart : Nat)
    (leftInputs : Assumptions leftInterface leftStart)
    (rightInputs : Assumptions rightInterface rightStart)
    (source : ∀ lane, (leftInterface.source lane leftStart).eval left =
      (rightInterface.source lane rightStart).eval right)
    (index : Nat) (bounded : index < WideReduction.privateCount) :
    Program.completeEnv leftInterface left leftStart (Program.coreOffset leftStart + index) =
      Program.completeEnv rightInterface right rightStart (Program.coreOffset rightStart + index) := by
  let leftChildren := Program.childEnv leftInterface left leftStart
  let rightChildren := Program.childEnv rightInterface right rightStart
  have cells := childEnv_local_congr leftInterface rightInterface left right leftStart rightStart
    leftInputs rightInputs source
  rw [completeEnv_reference _ _ _ leftInputs, completeEnv_reference _ _ _ rightInputs]
  by_cases child : index < childWidth * fieldCount
  · rw [show completeNew (Program.coreInterface leftInterface leftStart) leftChildren
        (Program.coreOffset leftStart) (Program.coreOffset leftStart + index) =
        leftChildren (Program.coreOffset leftStart + index) from
        completeNew_agreesOutside _ _ _ _ (Or.inl (by unfold quotientStart; omega)),
      show completeNew (Program.coreInterface rightInterface rightStart) rightChildren
        (Program.coreOffset rightStart) (Program.coreOffset rightStart + index) =
        rightChildren (Program.coreOffset rightStart + index) from
        completeNew_agreesOutside _ _ _ _ (Or.inl (by unfold quotientStart; omega))]
    have lane : index / childWidth < fieldCount := (Nat.div_lt_iff_lt_mul (by decide)).mpr child
    have position : index % childWidth < CanonicalU64.auxiliaryCount := Nat.mod_lt _ (by decide)
    have split := Nat.mod_add_div index childWidth
    convert cells ⟨index / childWidth, lane⟩ (index % childWidth) position using 1 <;>
      congr 1 <;> simp only [childOffset] <;> omega
  · have drawSame : drawOf (Program.coreInterface leftInterface leftStart) leftChildren (Program.coreOffset leftStart) =
        drawOf (Program.coreInterface rightInterface rightStart) rightChildren (Program.coreOffset rightStart) := by
      funext lane
      exact (childEnv_source _ _ _ leftInputs lane).trans
        ((source lane).trans (childEnv_source _ _ _ rightInputs lane).symm)
    have checks := checkQuotients_congr leftChildren rightChildren
      (Program.coreOffset leftStart) (Program.coreOffset rightStart)
      (Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.drawIndex
        (drawOf (Program.coreInterface rightInterface rightStart) rightChildren (Program.coreOffset rightStart))).val
      (fun lane bit below => cells lane bit (lt_trans below (by decide)))
    unfold completeNew
    rw [drawSame, checks]
    simp only [completedEnv, quotientStart, if_pos (show
      Program.coreOffset leftStart + childWidth * fieldCount ≤ Program.coreOffset leftStart + index ∧
        Program.coreOffset leftStart + index < Program.coreOffset leftStart + childWidth * fieldCount + newBitCount from
      by change index < childWidth * fieldCount + newBitCount at bounded; constructor <;> omega),
      if_pos (show Program.coreOffset rightStart + childWidth * fieldCount ≤ Program.coreOffset rightStart + index ∧
        Program.coreOffset rightStart + index < Program.coreOffset rightStart + childWidth * fieldCount + newBitCount from
      by change index < childWidth * fieldCount + newBitCount at bounded; constructor <;> omega),
      Nat.add_sub_add_left]
    rfl

end NightstreamFPrime.Gadgets.Sampling.WideReduction.ProgramValues

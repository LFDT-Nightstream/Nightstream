import NightstreamFPrime.Export.Stage1.PiCCSPoseidonPlan
import NightstreamFPrime.Export.Stage1.PiRLCRetainedPreservation
import NightstreamFPrime.Export.Stage1.PoseidonActionSemantics

/-!
Owns the value-preservation bridge for the direct PiCCS Poseidon2 plan.
It proves that every sparse invocation input is the exact previous output plus
the Lean action payload, with zero initial state and unchanged squeeze input.

The direct PiCCS plan owns the two squeeze-expectation pin rows.
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSPoseidonPreservation

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Poseidon2.Duplex
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation

def sourceAssignment (program : Lifecycle.Stage1.Application.Program)
    (prefixAssignment :
      Fin (PiCCSActionPayloadBlock.prefixSourceWidth program) → F) :
    Fin (PiCCSPoseidonPlan.sourceWidth program) → F :=
  PiCCSActionPayloadBlock.sourceAssignment program prefixAssignment

theorem sourceToSpartan_lt_basePackage (column : Nat)
    (bound : column < NightstreamFPrime.Layout.Stage1.Spartan.SourceColumnCount) :
    NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan column <
      PiRLCProductPlan.basePackage.layout.totalColumnCount := by
  have mapped := NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan_lt
    column bound
  rw [NightstreamFPrime.Layout.Stage1.Spartan.spartanColumnCount_eq] at mapped
  simpa [PiRLCProductPlan.basePackage] using mapped

def logicalPackageColumn (program : Lifecycle.Stage1.Application.Program)
    (column : Nat)
    (bound : column < NightstreamFPrime.Layout.Stage1.Spartan.SourceColumnCount) :
    Fin (PiRLCProductPlan.baseSourceWidth program) :=
  PiRLCProductPlan.shiftedPackageColumn program
    (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan column)
    (sourceToSpartan_lt_basePackage column bound)

/-- PiCCS payload expressions read every shifted base-package column from the
base assignment. PiRLC-derived suffixes cannot alias private or public inputs. -/
theorem packageEnv_sourceAssignment
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (column : Nat)
    (bound : column <
      NightstreamFPrime.Layout.Stage1.Spartan.SourceColumnCount) :
    PiCCSActionPayloadBlock.packageEnv program
        (PiRLCRetainedPreservation.sourceAssignment
          program base groupValue products) column =
      base (logicalPackageColumn program column bound) := by
  let mapped := NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan column
  let mappedBound := sourceToSpartan_lt_basePackage column bound
  have shiftedBase : PerApplicationPackage.shiftColumn program mapped <
      PiRLCProductPlan.baseSourceWidth program :=
    PiRLCProductPlan.shiftColumn_lt_baseSourceWidth program mapped mappedBound
  have shiftedPrefix :
      PerApplicationPackage.shiftColumn program mapped <
        PiCCSActionPayloadBlock.prefixSourceWidth program := by
    unfold PiCCSActionPayloadBlock.prefixSourceWidth
      PiRLCRetainedGeometry.sourceWidth
      PiRLCFirst54DirectPlan.sourceWidth
      PiRLCFirst54DirectPlan.prefixSourceWidth
      PiRLCProductPlan.sourceWidth ProductRetainedBlock.sourceWidth
      FieldSuffixBlock.sourceWidth
    omega
  unfold PiCCSActionPayloadBlock.packageEnv
    NightstreamFPrime.Layout.Stage1.Spartan.pullback
    PerApplicationPackage.baseEnv SourceCompiler.sourceEnv
  rw [dif_pos shiftedPrefix]
  rw [show
      (⟨PerApplicationPackage.shiftColumn program mapped, shiftedPrefix⟩ :
        Fin (PiCCSActionPayloadBlock.prefixSourceWidth program)) =
        PiRLCRetainedPreservation.baseSourceColumn program
          (PiRLCProductPlan.shiftedPackageColumn program mapped mappedBound) by
    apply Fin.ext
    rfl]
  rw [PiRLCRetainedPreservation.sourceAssignment_base]
  rfl

structure Encoding {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (prefixAssignment :
      Fin (PiCCSActionPayloadBlock.prefixSourceWidth program) → F) : Prop where
  payload : (PiCCSActionPayloadBlock.block program).EncodesAt
    (PiCCSActionPayloadBlock.payloadStart program)
    (PiCCSPoseidonPlan.payloadFits geometry) assignment
    (sourceAssignment program prefixAssignment)
  sboxes : (PiCCSPoseidonPlan.retainedBlock program).EncodesAt
    (PiCCSPoseidonPlan.retainedStart program)
    (PiCCSPoseidonPlan.retainedFits geometry) assignment
    (sourceAssignment program prefixAssignment)

/-- The canonical parent retained block supplies the PiCCS S-box slice. The
payload block is the only additional encoding obligation. -/
theorem encodingOfRetained
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (prefixAssignment :
      Fin (PiCCSActionPayloadBlock.prefixSourceWidth program) → F)
    (parentFits : PiRLCRetainedGeometry.laterPoseidonStart program +
      (PiRLCRetainedGeometry.laterPoseidonBlock program).coordinateCount ≤
        logicalWidth)
    (retained : (PiRLCRetainedGeometry.laterPoseidonBlock program).EncodesAt
      (PiRLCRetainedGeometry.laterPoseidonStart program) parentFits assignment
      prefixAssignment)
    (payload : (PiCCSActionPayloadBlock.block program).EncodesAt
      (PiCCSActionPayloadBlock.payloadStart program)
      (PiCCSPoseidonPlan.payloadFits geometry) assignment
      (sourceAssignment program prefixAssignment)) :
    Encoding geometry assignment prefixAssignment := by
  refine ⟨payload, ?_⟩
  exact PiCCSPoseidonPlan.retainedBlock_encodesAt geometry assignment
    prefixAssignment parentFits retained

def payloadLaneValue (program : Lifecycle.Stage1.Application.Program)
    (prefixAssignment :
      Fin (PiCCSActionPayloadBlock.prefixSourceWidth program) → F)
    (invocation : Fin PiCCSPoseidonPlan.invocationCount)
    (lane : Fin Spec.Poseidon2.width) : F :=
  if rateLane : lane.val < Spec.Poseidon2.rate then
    PiCCSActionPayloadBlock.payloadValue program prefixAssignment
      (Fin.encodeProd (invocation, ⟨lane.val, rateLane⟩))
  else
    0

theorem payloadForm_eval
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (prefixAssignment :
      Fin (PiCCSActionPayloadBlock.prefixSourceWidth program) → F)
    (encoding : Encoding geometry assignment prefixAssignment)
    (invocation : Fin PiCCSPoseidonPlan.invocationCount)
    (lane : Fin Spec.Poseidon2.width) :
    (PiCCSPoseidonPlan.payloadForm geometry invocation lane).eval assignment =
      payloadLaneValue program prefixAssignment invocation lane := by
  unfold PiCCSPoseidonPlan.payloadForm payloadLaneValue
  split
  · rw [LowNormBlock.Block.form_eval]
    · exact PiCCSActionPayloadBlock.block_sourceAssignment
        program prefixAssignment _
    · exact encoding.payload
  · exact SparseForm.empty_eval assignment

theorem payloadLaneValue_absorb
    (program : Lifecycle.Stage1.Application.Program)
    (prefixAssignment :
      Fin (PiCCSActionPayloadBlock.prefixSourceWidth program) → F)
    (invocation : Fin PiCCSPoseidonPlan.invocationCount)
    (lane : Fin Spec.Poseidon2.width) (block : List Circuit.Expr)
    (found : PiCCSActionPayloadBlock.kindAt invocation = .absorb block) :
    payloadLaneValue program prefixAssignment invocation lane =
      (Hash.evalList
        (PiCCSActionPayloadBlock.packageEnv program prefixAssignment) block).getD
          lane.val 0 := by
  unfold payloadLaneValue
  by_cases rateLane : lane.val < Spec.Poseidon2.rate
  · rw [dif_pos rateLane]
    unfold PiCCSActionPayloadBlock.payloadValue
    rw [PiCCSActionPayloadBlock.payloadExpression_encode]
    unfold PiCCSActionPayloadBlock.payloadExpr
      PiCCSActionPayloadBlock.selectedBlock
    rw [found]
    change (block.getD lane.val (0 : Circuit.Expr)).eval
        (PiCCSActionPayloadBlock.packageEnv program prefixAssignment) = _
    have evalZero : (0 : Circuit.Expr).eval
        (PiCCSActionPayloadBlock.packageEnv program prefixAssignment) =
        (0 : F) := by
      apply Fin.ext
      norm_num [Circuit.Expr.eval, goldilocksModulus]
    rw [← evalZero]
    exact (List.getD_map (n := lane.val) block (0 : Circuit.Expr)
      (Circuit.Expr.eval
        (PiCCSActionPayloadBlock.packageEnv program prefixAssignment))).symm
  · rw [dif_neg rateLane]
    have wellFormed := PiCCSActionPayloadBlock.kindAt_wellFormed invocation
    rw [found] at wellFormed
    change block.length ≤ Spec.Poseidon2.rate at wellFormed
    apply Eq.symm
    apply List.getD_eq_default
    simp [Hash.evalList]
    omega

theorem payloadLaneValue_squeezeFirst_zero
    (program : Lifecycle.Stage1.Application.Program)
    (prefixAssignment :
      Fin (PiCCSActionPayloadBlock.prefixSourceWidth program) → F)
    (invocation : Fin PiCCSPoseidonPlan.invocationCount)
    (expected : NightstreamFPrime.Circuit.Quadratic.KExpr)
    (found : PiCCSActionPayloadBlock.kindAt invocation =
      .squeezeFirst expected) :
    payloadLaneValue program prefixAssignment invocation (0 : Fin 8) =
      expected.c0.eval
        (PiCCSActionPayloadBlock.packageEnv program prefixAssignment) := by
  unfold payloadLaneValue PiCCSActionPayloadBlock.payloadValue
  rw [dif_pos (by norm_num [Spec.Poseidon2.rate])]
  rw [PiCCSActionPayloadBlock.payloadExpression_encode]
  unfold PiCCSActionPayloadBlock.payloadExpr
    PiCCSActionPayloadBlock.selectedBlock
  rw [found]
  rfl

theorem payloadLaneValue_squeezeFirst_one
    (program : Lifecycle.Stage1.Application.Program)
    (prefixAssignment :
      Fin (PiCCSActionPayloadBlock.prefixSourceWidth program) → F)
    (invocation : Fin PiCCSPoseidonPlan.invocationCount)
    (expected : NightstreamFPrime.Circuit.Quadratic.KExpr)
    (found : PiCCSActionPayloadBlock.kindAt invocation =
      .squeezeFirst expected) :
    payloadLaneValue program prefixAssignment invocation (1 : Fin 8) =
      expected.c1.eval
        (PiCCSActionPayloadBlock.packageEnv program prefixAssignment) := by
  unfold payloadLaneValue PiCCSActionPayloadBlock.payloadValue
  rw [dif_pos (by norm_num [Spec.Poseidon2.rate])]
  rw [PiCCSActionPayloadBlock.payloadExpression_encode]
  unfold PiCCSActionPayloadBlock.payloadExpr
    PiCCSActionPayloadBlock.selectedBlock
  rw [found]
  rfl

def outputValue {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (invocation : Fin PiCCSPoseidonPlan.invocationCount) :
    NightstreamFPrime.Gadgets.Poseidon2.Layer.FState :=
  SparseLayer.evalState assignment
    ((PiCCSPoseidonPlan.interface geometry).output invocation)

/-- Every retained final-state lane reconstructs the exact canonical source
value selected by the PiCCS Poseidon schedule. -/
theorem outputValue_sourceAssignment
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (prefixAssignment :
      Fin (PiCCSActionPayloadBlock.prefixSourceWidth program) → F)
    (encoding : Encoding geometry assignment prefixAssignment)
    (invocation : Fin PiCCSPoseidonPlan.invocationCount) :
    outputValue geometry assignment invocation =
      NightstreamFPrime.Gadgets.Poseidon2.Layer.externalF (fun lane =>
        sourceAssignment program prefixAssignment
          ((PiCCSPoseidonPlan.schedule program).block.source
            (PoseidonRetainedFamily.slot (PiCCSPoseidonPlan.schedule program)
              invocation (PoseidonRetainedSlots.finalRow lane)))) := by
  exact PoseidonRetainedFamily.outputState_eval
    (PiCCSPoseidonPlan.schedule program)
    (PiCCSPoseidonPlan.retainedStart program)
    (PiCCSPoseidonPlan.retainedFits geometry) assignment
    (sourceAssignment program prefixAssignment) encoding.sboxes invocation

def previousValue {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (invocation : Fin PiCCSPoseidonPlan.invocationCount) :
    NightstreamFPrime.Gadgets.Poseidon2.Layer.FState :=
  if first : invocation.val = 0 then
    fun _ => 0
  else
    outputValue geometry assignment
      ⟨invocation.val - 1, by
        have invocationBound := invocation.isLt
        omega⟩

theorem previousOutput_eval
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (invocation : Fin PiCCSPoseidonPlan.invocationCount) :
    SparseLayer.evalState assignment
        (PiCCSPoseidonPlan.previousOutput geometry invocation) =
      previousValue geometry assignment invocation := by
  funext lane
  unfold PiCCSPoseidonPlan.previousOutput previousValue outputValue
  split
  · exact SparseForm.empty_eval assignment
  · rfl

def canonicalInput {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (prefixAssignment :
      Fin (PiCCSActionPayloadBlock.prefixSourceWidth program) → F)
    (invocation : Fin PiCCSPoseidonPlan.invocationCount) :
    NightstreamFPrime.Gadgets.Poseidon2.Layer.FState :=
  let previous := previousValue geometry assignment invocation
  match PiCCSActionPayloadBlock.kindAt invocation with
  | .absorb _ => fun lane =>
      previous lane +
        payloadLaneValue program prefixAssignment invocation lane
  | .squeezeFirst _ => previous
  | .squeezeSecond => previous

theorem inputState_eval
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (prefixAssignment :
      Fin (PiCCSActionPayloadBlock.prefixSourceWidth program) → F)
    (encoding : Encoding geometry assignment prefixAssignment)
    (invocation : Fin PiCCSPoseidonPlan.invocationCount) :
    SparseLayer.evalState assignment
        (PiCCSPoseidonPlan.inputState geometry invocation) =
      canonicalInput geometry assignment prefixAssignment invocation := by
  funext lane
  unfold PiCCSPoseidonPlan.inputState canonicalInput
  cases found : PiCCSActionPayloadBlock.kindAt invocation with
  | absorb block =>
      simp only [SparseLayer.evalState, SparseForm.add_eval]
      rw [show
          (PiCCSPoseidonPlan.previousOutput geometry invocation lane).eval
              assignment =
            previousValue geometry assignment invocation lane by
        exact congrFun
          (previousOutput_eval geometry assignment invocation) lane]
      rw [payloadForm_eval geometry assignment prefixAssignment encoding]
  | squeezeFirst expected =>
      simp only [SparseLayer.evalState]
      exact congrFun (previousOutput_eval geometry assignment invocation) lane
  | squeezeSecond =>
      simp only [SparseLayer.evalState]
      exact congrFun (previousOutput_eval geometry assignment invocation) lane

structure CanonicalSemantics
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (prefixAssignment :
      Fin (PiCCSActionPayloadBlock.prefixSourceWidth program) → F) : Prop where
  invocation : ∀ current,
    List.ofFn (outputValue geometry assignment current) =
      Spec.Poseidon2.permute
        (List.ofFn
          (canonicalInput geometry assignment prefixAssignment current))
  squeezeExpected : ∀ current expected,
    PiCCSActionPayloadBlock.kindAt current = .squeezeFirst expected →
      expected.eval (PiCCSActionPayloadBlock.packageEnv program prefixAssignment) =
        ⟨previousValue geometry assignment current 0,
          outputValue geometry assignment current 0⟩

def valueState {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (current : Fin PiCCSActionPayloadBlock.invocationCount) :
    Spec.Poseidon2.State :=
  List.ofFn (outputValue geometry assignment current)

theorem previousState_eq_previousValue
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (current : Fin PiCCSActionPayloadBlock.invocationCount) :
    PoseidonActionSemantics.previousState Spec.Poseidon2.zeroState
        (valueState geometry assignment) current =
      List.ofFn (previousValue geometry assignment current) := by
  unfold PoseidonActionSemantics.previousState previousValue valueState
  by_cases first : current.val = 0
  · rw [dif_pos first, dif_pos first]
    unfold Spec.Poseidon2.zeroState
    change List.replicate 8 0 = List.ofFn (fun _ : Fin 8 => (0 : F))
    norm_num [List.ofFn_succ]
    rfl
  · rw [dif_neg first, dif_neg first]
    apply congrArg (fun index =>
      List.ofFn (outputValue geometry assignment index))
    apply Fin.ext
    rfl

def indexedSemantics
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (prefixAssignment :
      Fin (PiCCSActionPayloadBlock.prefixSourceWidth program) → F)
    (semantics : CanonicalSemantics geometry assignment prefixAssignment) :
    PoseidonActionSemantics.IndexedSemantics
      (PiCCSActionPayloadBlock.packageEnv program prefixAssignment)
      Spec.Poseidon2.zeroState PiCCSActionPayloadBlock.kindAt
      (valueState geometry assignment) where
  step current := by
    have previousEq :
        PoseidonActionSemantics.previousState Spec.Poseidon2.zeroState
            (valueState geometry assignment) current =
          List.ofFn (previousValue geometry assignment current) := by
      exact previousState_eq_previousValue geometry assignment current
    rw [previousEq]
    have invocation := semantics.invocation current
    cases found : PiCCSActionPayloadBlock.kindAt current with
    | absorb block =>
        have inputEq :
            canonicalInput geometry assignment prefixAssignment current =
              Hash.absorbF (previousValue geometry assignment current)
                (Hash.evalList
                  (PiCCSActionPayloadBlock.packageEnv program prefixAssignment)
                  block) := by
          funext lane
          unfold canonicalInput Hash.absorbF
          rw [found]
          exact congrArg
            (fun value => previousValue geometry assignment current lane + value)
            (payloadLaneValue_absorb program prefixAssignment current lane block
              found)
        unfold valueState PoseidonActionSemantics.runKind
        rw [inputEq] at invocation
        calc
          List.ofFn (outputValue geometry assignment current) =
              Spec.Poseidon2.permute
                (List.ofFn (Hash.absorbF
                  (previousValue geometry assignment current)
                  (Hash.evalList
                    (PiCCSActionPayloadBlock.packageEnv program prefixAssignment)
                    block))) := invocation
          _ = Spec.Poseidon2.absorbBlock
                (List.ofFn (previousValue geometry assignment current))
                (Hash.evalList
                  (PiCCSActionPayloadBlock.packageEnv program prefixAssignment)
                  block) := by
            rw [Spec.Poseidon2.absorbBlock,
              Hash.absorbF_input_eq_reference]
    | squeezeFirst expected =>
        simpa [valueState, PoseidonActionSemantics.runKind, canonicalInput,
          found] using invocation
    | squeezeSecond =>
        simpa [valueState, PoseidonActionSemantics.runKind, canonicalInput,
          found] using invocation
  expected current expected found := by
    have previousEq :
        PoseidonActionSemantics.previousState Spec.Poseidon2.zeroState
            (valueState geometry assignment) current =
          List.ofFn (previousValue geometry assignment current) := by
      exact previousState_eq_previousValue geometry assignment current
    rw [previousEq]
    have bound := semantics.squeezeExpected current expected found
    have invocation := semantics.invocation current
    have permuteEq : valueState geometry assignment current =
        Spec.Poseidon2.permute
          (List.ofFn (previousValue geometry assignment current)) := by
      simpa [valueState, canonicalInput, found] using invocation
    unfold Squeeze.referenceSample
    rw [bound]
    apply congrArg₂ K.mk
    · exact (NightstreamFPrime.Lifecycle.PriorStateHash.ofFn_getD
        (previousValue geometry assignment current) 0 0).symm
    · rw [← permuteEq]
      exact (NightstreamFPrime.Lifecycle.PriorStateHash.ofFn_getD
        (outputValue geometry assignment current) 0 0).symm

theorem squeezeExpected_eval
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (prefixAssignment :
      Fin (PiCCSActionPayloadBlock.prefixSourceWidth program) → F)
    (encoding : Encoding geometry assignment prefixAssignment)
    (semantics : PiCCSPoseidonPlan.Semantics geometry assignment)
    (current : Fin PiCCSPoseidonPlan.invocationCount)
    (expected : NightstreamFPrime.Circuit.Quadratic.KExpr)
    (found : PiCCSActionPayloadBlock.kindAt current = .squeezeFirst expected) :
    expected.eval (PiCCSActionPayloadBlock.packageEnv program prefixAssignment) =
      ⟨previousValue geometry assignment current 0,
        outputValue geometry assignment current 0⟩ := by
  have rowZero := semantics.squeezeBinding current (0 : Fin 2)
  rw [PiCCSPoseidonPlan.bindingForm_squeezeFirst_zero geometry current
    expected found, SparseForm.add_eval, SparseForm.scale_eval] at rowZero
  rw [payloadForm_eval geometry assignment prefixAssignment encoding] at rowZero
  rw [show
      (PiCCSPoseidonPlan.previousOutput geometry current 0).eval assignment =
        previousValue geometry assignment current 0 by
    exact congrFun (previousOutput_eval geometry assignment current) 0] at rowZero
  have c0 : expected.c0.eval
      (PiCCSActionPayloadBlock.packageEnv program prefixAssignment) =
      previousValue geometry assignment current 0 := by
    apply Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp
    simpa [sub_eq_add_neg,
      payloadLaneValue_squeezeFirst_zero program prefixAssignment current
        expected found] using rowZero
  have rowOne := semantics.squeezeBinding current (1 : Fin 2)
  rw [PiCCSPoseidonPlan.bindingForm_squeezeFirst_one geometry current
    expected found, SparseForm.add_eval, SparseForm.scale_eval] at rowOne
  rw [payloadForm_eval geometry assignment prefixAssignment encoding] at rowOne
  have c1 : expected.c1.eval
      (PiCCSActionPayloadBlock.packageEnv program prefixAssignment) =
      outputValue geometry assignment current 0 := by
    apply Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp
    simpa [sub_eq_add_neg,
      payloadLaneValue_squeezeFirst_one program prefixAssignment current
        expected found, PiCCSPoseidonPlan.outputState, outputValue,
      PiCCSPoseidonPlan.interface] using rowOne
  exact congrArg₂ NightstreamFPrime.Spec.K.mk c0 c1

theorem rowsZero_implies_canonicalSemantics
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (prefixAssignment :
      Fin (PiCCSActionPayloadBlock.prefixSourceWidth program) → F)
    (encoding : Encoding geometry assignment prefixAssignment)
    (one : assignment (PiCCSPoseidonPlan.oneColumn geometry) = 1)
    (rowsZero : (PiCCSPoseidonPlan.plan geometry).RowsZero assignment) :
    CanonicalSemantics geometry assignment prefixAssignment := by
  have semantics := PiCCSPoseidonPlan.rowsZero_implies_semantics
    geometry assignment one rowsZero
  refine ⟨?_, ?_⟩
  intro invocation
  calc
    List.ofFn (outputValue geometry assignment invocation) =
        Spec.Poseidon2.permute
          (List.ofFn (SparseLayer.evalState assignment
            ((PiCCSPoseidonPlan.interface geometry).input invocation))) :=
      semantics.invocation invocation
    _ = Spec.Poseidon2.permute
          (List.ofFn
            (canonicalInput geometry assignment prefixAssignment invocation)) := by
      change Spec.Poseidon2.permute
          (List.ofFn (SparseLayer.evalState assignment
            (PiCCSPoseidonPlan.inputState geometry invocation))) = _
      rw [inputState_eval geometry assignment prefixAssignment encoding]
  · intro current expected found
    exact squeezeExpected_eval geometry assignment prefixAssignment encoding
      semantics current expected found

end NightstreamFPrime.Export.Stage1.PiCCSPoseidonPreservation

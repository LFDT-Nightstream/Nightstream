import NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.StatementAbsorption
import NightstreamFPrime.Spec.Folding.PiCCS.Accepted

/-!
Paper authority: SuperNeo v1.2, Section 7.3, Steps 3 and 5; the
Fiat–Shamir handoff to `Pi_RLC`.
Obligation: Reuse the complete prover `y′` family as the 17 reduced CE
evaluation families, absorb it in canonical source/Pad/matrix order, and
bind the verifier-owned outgoing transcript state.

Inputs:
- the verifier-derived final SumCheck point and transcript state;
- all 17 separate Pad (`Eval_K`) output families;
- all 17 × 4 CCS-matrix (`Eval_A`) output families.

Outputs:
- zero-copy reduced point, `Eval_K`, and `Eval_A` views;
- the transcript state from which `Pi_RLC` samples all 17 challenges.

Constraint groups:
- C1: one complete output absorption, with no length prefix, through the
  generic Duplex circuit;
- C2: no final-state or reduced-claim copy rows.

Parent coverage:
- `v1_2.Coverage.output_eval_K`;
- `v1_2.Coverage.output_eval_A`;
- `Key.piCcsExecution.outgoingState`.
-/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_2.OutputBinding

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Poseidon2.Duplex
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources

/-- The output expressions for a Π_CCS shape: the production shape, or the bounded-depth scheme's light shape. -/
structure ShapeOutputExpr (shape : Shape) where
  padCoordinate : Fin shape.sourceCount →
    Fin shape.coefficientCount → KExpr
  matrixCoordinate : Fin shape.sourceCount →
    Fin shape.matrixCount →
    Fin shape.coefficientCount → KExpr

abbrev OutputExpr := ShapeOutputExpr productionShape

structure ShapeInterface (shape : Shape) where
  roundPoint : Nat → Fin shape.cubeVariables → KExpr
  initialState : Nat → Layer.EState
  output : Nat → ShapeOutputExpr shape

abbrev Interface := ShapeInterface productionShape

section Shape

variable {shape : Shape}

def reducedPoint (interface : ShapeInterface shape) (offset : Nat)
    (_source : Fin shape.sourceCount) :=
  interface.roundPoint offset

def reducedEval_K (interface : ShapeInterface shape) (offset : Nat)
    (source : Fin shape.sourceCount)
    (coefficient : Fin shape.coefficientCount) : KExpr :=
  (interface.output offset).padCoordinate source coefficient

def reducedEval_A (interface : ShapeInterface shape) (offset : Nat)
    (source : Fin shape.sourceCount)
    (matrix : Fin shape.matrixCount)
    (coefficient : Fin shape.coefficientCount) : KExpr :=
  (interface.output offset).matrixCoordinate source matrix coefficient

@[simp] theorem reducedPoint_eq (interface : ShapeInterface shape) (offset : Nat)
    (source : Fin shape.sourceCount) :
    reducedPoint interface offset source = interface.roundPoint offset := by
  rfl

@[simp] theorem reducedEval_K_eq (interface : ShapeInterface shape) (offset : Nat)
    (source : Fin shape.sourceCount)
    (coefficient : Fin shape.coefficientCount) :
    reducedEval_K interface offset source coefficient =
      (interface.output offset).padCoordinate source coefficient := by
  rfl

@[simp] theorem reducedEval_A_eq (interface : ShapeInterface shape) (offset : Nat)
    (source : Fin shape.sourceCount)
    (matrix : Fin shape.matrixCount)
    (coefficient : Fin shape.coefficientCount) :
    reducedEval_A interface offset source matrix coefficient =
      (interface.output offset).matrixCoordinate source matrix coefficient := by
  rfl

def padWords (output : ShapeOutputExpr shape)
    (source : Fin shape.sourceCount) : List Expr :=
  (List.finRange shape.coefficientCount).flatMap fun coefficient =>
    StatementAbsorption.serializeKExpr
      (output.padCoordinate source coefficient)

def matrixWords (output : ShapeOutputExpr shape)
    (source : Fin shape.sourceCount) : List Expr :=
  (List.finRange shape.matrixCount).flatMap fun matrix =>
    (List.finRange shape.coefficientCount).flatMap fun coefficient =>
      StatementAbsorption.serializeKExpr
        (output.matrixCoordinate source matrix coefficient)

def sourceWords (output : ShapeOutputExpr shape)
    (source : Fin shape.sourceCount) : List Expr :=
  padWords output source ++ matrixWords output source

def outputWords (interface : ShapeInterface shape) (offset : Nat) : List Expr :=
  (List.finRange shape.sourceCount).flatMap fun source =>
    sourceWords (interface.output offset) source

end Shape

private theorem padWords_length (output : OutputExpr)
    (source : Fin productionShape.sourceCount) :
    (padWords output source).length = 108 := by
  unfold padWords
  rw [flatMap_length_constant _ _ 2 (fun _ => rfl)]
  norm_num [productionShape, productionProfile,
    RingMatrixSource.ringShape, ringDegree]

private theorem matrixWords_length (output : OutputExpr)
    (source : Fin productionShape.sourceCount) :
    (matrixWords output source).length = 432 := by
  have coefficientLength : ∀ matrix,
      ((List.finRange productionShape.coefficientCount).flatMap
        fun coefficient => StatementAbsorption.serializeKExpr
          (output.matrixCoordinate source matrix coefficient)).length = 108 := by
    intro matrix
    rw [flatMap_length_constant _ _ 2 (fun _ => rfl)]
    norm_num [productionShape, productionProfile,
      RingMatrixSource.ringShape, ringDegree]
  unfold matrixWords
  rw [flatMap_length_constant _ _ 108 coefficientLength]
  norm_num [productionShape, productionProfile,
    RingMatrixSource.ringShape, ringDegree]

private theorem sourceWords_length (output : OutputExpr)
    (source : Fin productionShape.sourceCount) :
    (sourceWords output source).length = 540 := by
  rw [sourceWords, List.length_append, padWords_length, matrixWords_length]

theorem outputWords_length (interface : Interface) (offset : Nat) :
    (outputWords interface offset).length = 9180 := by
  unfold outputWords
  rw [flatMap_length_constant _ _ 540 (sourceWords_length _)]
  norm_num [productionShape, productionProfile,
    RingMatrixSource.ringShape, Shape.sourceCount]

section Shape

variable {shape : Shape}

def actions (interface : ShapeInterface shape) (offset : Nat) : List Formal.Action :=
  [.absorb (outputWords interface offset)]

def duplexInterface (interface : ShapeInterface shape) : Formal.Owned.Interface where
  initial := interface.initialState
  actions := actions interface

def finalState (interface : ShapeInterface shape) (offset : Nat) : Layer.EState :=
  Formal.Owned.output (duplexInterface interface) offset

theorem finalState_eq_compile (interface : ShapeInterface shape) (offset : Nat) :
    finalState interface offset =
      (Formal.compile offset (interface.initialState offset)
        (actions interface offset)).output := by
  rfl

abbrev Assumptions (interface : ShapeInterface shape) (offset : Nat) (env : Env) : Prop :=
  Formal.Owned.Assumptions (duplexInterface interface) offset env

abbrev SpecHolds (interface : ShapeInterface shape) (offset : Nat) (env : Env) : Prop :=
  Formal.Owned.SpecHolds (duplexInterface interface) offset env

theorem trace_implies_specHolds (interface : ShapeInterface shape) (offset : Nat)
    (env : Env)
    (trace : Formal.TraceHolds
      (List.ofFn (Layer.evalState env (interface.initialState offset)))
      ((actions interface offset).map (Formal.Action.eval env))
      (List.ofFn (Layer.evalState env (finalState interface offset)))) :
    SpecHolds interface offset env :=
  trace

def circuit (interface : ShapeInterface shape) : FormalCircuit :=
  Formal.Owned.circuit (duplexInterface interface)

theorem soundness (interface : ShapeInterface shape) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (circuit interface).main offset)) :
    SpecHolds interface offset env :=
  Formal.Owned.soundness (duplexInterface interface) env offset assumptions rows

theorem noAssertions (interface : ShapeInterface shape) (offset : Nat) :
    Formal.Owned.allAssertions (duplexInterface interface) offset = [] := by
  rfl

theorem build (interface : ShapeInterface shape) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (circuit interface).main offset)) ∧
      holdsFlat completed (Circuit.ops (circuit interface).main offset) :=
  Formal.Owned.build_of_no_assertions (duplexInterface interface) env offset
    assumptions (noAssertions interface offset)

theorem completeness (interface : ShapeInterface shape) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (circuit interface).main offset)) ∧
      holdsFlat completed (Circuit.ops (circuit interface).main offset) :=
  Formal.Owned.completeness (duplexInterface interface) env offset assumptions
    specification

theorem flatConstraints_varsBelow (interface : ShapeInterface shape) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints
      (Circuit.ops (circuit interface).main offset),
      expression.VarsBelow
        (offset + localLength (Circuit.ops (circuit interface).main offset)) :=
  Formal.Owned.flatConstraints_varsBelow (duplexInterface interface) offset env
    assumptions

theorem finalState_varsBelow (interface : ShapeInterface shape) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    ∀ lane, (finalState interface offset lane).VarsBelow
      (offset + localLength (Circuit.ops (circuit interface).main offset)) :=
  Formal.Owned.output_varsBelow (duplexInterface interface) offset env
    assumptions

theorem actions_length (interface : ShapeInterface shape) (offset : Nat) :
    (actions interface offset).length = 1 := by
  rfl

end Shape

def recipeCount (interface : Interface) (offset : Nat) : Nat :=
  Formal.recipeCount (actions interface offset)

private theorem inputChunks_length (input : List Expr) :
    (Hash.inputChunks input).length = (input.length + 11) / 12 := by
  unfold Hash.inputChunks
  rw [List.length_map, List.length_range]
  rfl

theorem recipeCount_eq (interface : Interface) (offset : Nat) :
    recipeCount interface offset = 838440 := by
  unfold recipeCount actions
  simp only [Formal.recipeCount, List.map_cons, List.map_nil,
    List.sum_cons, List.sum_nil, Nat.add_zero, Formal.Action.recipeCount]
  rw [inputChunks_length, outputWords_length]

theorem localLength_eq (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (circuit interface).main offset) = 838440 := by
  change localLength (Formal.Owned.opsAt (duplexInterface interface) offset) = _
  rw [Formal.Owned.opsAt_localLength]
  unfold Formal.Owned.program
  rw [Formal.compile_recipes_length]
  simpa [duplexInterface, recipeCount] using recipeCount_eq interface offset

theorem operations_length (interface : Interface) (offset : Nat) :
    (Circuit.ops (circuit interface).main offset).length = 1 := by
  change (Formal.Owned.opsAt (duplexInterface interface) offset).length = 1
  rw [Formal.Owned.operations_length]
  simp [duplexInterface, actions, Formal.assertionCount,
    Formal.Action.assertionCount]

theorem flatConstraints_length (interface : Interface) (offset : Nat) :
    (flatConstraints (Circuit.ops (circuit interface).main offset)).length =
      838440 := by
  change (flatConstraints
    (Formal.Owned.opsAt (duplexInterface interface) offset)).length = _
  rw [Formal.Owned.flatConstraints_length]
  have recipes : Formal.recipeCount
      ((duplexInterface interface).actions offset) = 838440 := by
    simpa [duplexInterface, recipeCount] using recipeCount_eq interface offset
  rw [recipes]
  simp [duplexInterface, actions, Formal.assertionCount,
    Formal.Action.assertionCount]

section Shape

variable {shape : Shape}

def valuePadWords
    (output : FullOutputCoordinates.FullOutput K shape)
    (source : Fin shape.sourceCount) : List F :=
  (List.finRange shape.coefficientCount).flatMap fun coefficient =>
    NightstreamFPrime.Lifecycle.serializeK
      (output.padCoordinate source coefficient)

def valueMatrixWords
    (output : FullOutputCoordinates.FullOutput K shape)
    (source : Fin shape.sourceCount) : List F :=
  (List.finRange shape.matrixCount).flatMap fun matrix =>
    (List.finRange shape.coefficientCount).flatMap fun coefficient =>
      NightstreamFPrime.Lifecycle.serializeK
        (output.matrixCoordinate source matrix coefficient)

@[simp] private theorem serializeKExpr_eval (env : Env) (value : KExpr) :
    (StatementAbsorption.serializeKExpr value).map (Expr.eval env) =
      NightstreamFPrime.Lifecycle.serializeK (value.eval env) := by
  rfl

private theorem map_flatMap_congr
    {Index Left Right : Type}
    (indices : List Index)
    (left : Index → List Left)
    (right : Index → List Right)
    (transform : Left → Right)
    (each : ∀ index, (left index).map transform = right index) :
    (indices.flatMap left).map transform = indices.flatMap right := by
  induction indices with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      rw [List.flatMap_cons, List.map_append, List.flatMap_cons,
        each, inductionHypothesis]

private theorem padWords_eval (interface : ShapeInterface shape) (offset : Nat)
    (env : Env) (output : FullOutputCoordinates.FullOutput K shape)
    (padEq : ∀ source coefficient,
      ((interface.output offset).padCoordinate source coefficient).eval env =
        output.padCoordinate source coefficient)
    (source : Fin shape.sourceCount) :
    (padWords (interface.output offset) source).map (Expr.eval env) =
      valuePadWords output source := by
  unfold padWords valuePadWords
  apply map_flatMap_congr
  intro coefficient
  calc
    (StatementAbsorption.serializeKExpr
        ((interface.output offset).padCoordinate source coefficient)).map
          (Expr.eval env) =
        NightstreamFPrime.Lifecycle.serializeK
          (((interface.output offset).padCoordinate source coefficient).eval
            env) := serializeKExpr_eval env _
    _ = NightstreamFPrime.Lifecycle.serializeK
          (output.padCoordinate source coefficient) := by
      rw [padEq]

private theorem matrixWords_eval (interface : ShapeInterface shape) (offset : Nat)
    (env : Env) (output : FullOutputCoordinates.FullOutput K shape)
    (matrixEq : ∀ source matrix coefficient,
      ((interface.output offset).matrixCoordinate source matrix
        coefficient).eval env =
        output.matrixCoordinate source matrix coefficient)
    (source : Fin shape.sourceCount) :
    (matrixWords (interface.output offset) source).map (Expr.eval env) =
      valueMatrixWords output source := by
  unfold matrixWords valueMatrixWords
  apply map_flatMap_congr
  intro matrix
  apply map_flatMap_congr
  intro coefficient
  calc
    (StatementAbsorption.serializeKExpr
        ((interface.output offset).matrixCoordinate source matrix
          coefficient)).map (Expr.eval env) =
        NightstreamFPrime.Lifecycle.serializeK
          (((interface.output offset).matrixCoordinate source matrix
            coefficient).eval env) := serializeKExpr_eval env _
    _ = NightstreamFPrime.Lifecycle.serializeK
          (output.matrixCoordinate source matrix coefficient) := by
      rw [matrixEq]

private theorem sourceWords_eval (interface : ShapeInterface shape) (offset : Nat)
    (env : Env) (output : FullOutputCoordinates.FullOutput K shape)
    (padEq : ∀ source coefficient,
      ((interface.output offset).padCoordinate source coefficient).eval env =
        output.padCoordinate source coefficient)
    (matrixEq : ∀ source matrix coefficient,
      ((interface.output offset).matrixCoordinate source matrix
        coefficient).eval env =
        output.matrixCoordinate source matrix coefficient)
    (source : Fin shape.sourceCount) :
    (sourceWords (interface.output offset) source).map (Expr.eval env) =
      valuePadWords output source ++ valueMatrixWords output source := by
  rw [sourceWords, List.map_append, padWords_eval interface offset env output
    padEq, matrixWords_eval interface offset env output matrixEq]

theorem outputWords_eval_values (interface : ShapeInterface shape) (offset : Nat) (env : Env)
    (output : FullOutputCoordinates.FullOutput K shape)
    (padEq : ∀ source coefficient,
      ((interface.output offset).padCoordinate source coefficient).eval env =
        output.padCoordinate source coefficient)
    (matrixEq : ∀ source matrix coefficient,
      ((interface.output offset).matrixCoordinate source matrix
        coefficient).eval env =
        output.matrixCoordinate source matrix coefficient) :
    Hash.evalList env (outputWords interface offset) =
      (List.finRange shape.sourceCount).flatMap fun source =>
        valuePadWords output source ++ valueMatrixWords output source := by
  unfold Hash.evalList outputWords
  apply map_flatMap_congr
  intro source
  exact sourceWords_eval interface offset env output padEq matrixEq source

/-- The rows absorb every source's output words into the transcript state they start from
(SuperNeo v1.2 Section 7.3, Step 5). -/
theorem spec_implies_absorb (interface : ShapeInterface shape) (offset : Nat) (env : Env)
    (state : Poseidon2.State) (output : FullOutputCoordinates.FullOutput K shape)
    (initialEq : List.ofFn (Layer.evalState env (interface.initialState offset)) = state)
    (padEq : ∀ source coefficient,
      ((interface.output offset).padCoordinate source coefficient).eval env =
        output.padCoordinate source coefficient)
    (matrixEq : ∀ source matrix coefficient,
      ((interface.output offset).matrixCoordinate source matrix
        coefficient).eval env =
        output.matrixCoordinate source matrix coefficient)
    (specification : SpecHolds interface offset env) :
    List.ofFn (Layer.evalState env (finalState interface offset)) =
      NightstreamFPrime.Lifecycle.Transcript.absorb state
        ((List.finRange shape.sourceCount).flatMap fun source =>
          valuePadWords output source ++ valueMatrixWords output source) := by
  have trace := specification
  change Absorb.reference
      (List.ofFn (Layer.evalState env (interface.initialState offset)))
      (Hash.evalList env (outputWords interface offset)) =
    List.ofFn (Layer.evalState env (finalState interface offset)) at trace
  rw [outputWords_eval_values interface offset env output padEq matrixEq, initialEq] at trace
  exact trace.symm

end Shape

theorem outputWords_eval (interface : Interface) (offset : Nat) (env : Env)
    (output : FullOutputCoordinates.FullOutput K productionShape)
    (padEq : ∀ source coefficient,
      ((interface.output offset).padCoordinate source coefficient).eval env =
        output.padCoordinate source coefficient)
    (matrixEq : ∀ source matrix coefficient,
      ((interface.output offset).matrixCoordinate source matrix
        coefficient).eval env =
        output.matrixCoordinate source matrix coefficient) :
    Hash.evalList env (outputWords interface offset) = ProductionKey.fullOutputWords output := by
  unfold Hash.evalList outputWords ProductionKey.fullOutputWords
  apply map_flatMap_congr
  intro source
  exact sourceWords_eval interface offset env output padEq matrixEq source

theorem spec_implies_keyOutgoingState
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Running
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof (ProductionKey.degreeBound relation))
    (interface : Interface) (offset : Nat) (env : Env)
    (initialEq : List.ofFn (Layer.evalState env
      (interface.initialState offset)) =
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.finalState)
    (padEq : ∀ source coefficient,
      ((interface.output offset).padCoordinate source coefficient).eval env =
        proof.piCcsOutput.padCoordinate source coefficient)
    (matrixEq : ∀ source matrix coefficient,
      ((interface.output offset).matrixCoordinate source matrix
        coefficient).eval env =
        proof.piCcsOutput.matrixCoordinate source matrix coefficient)
    (specification : SpecHolds interface offset env) :
    List.ofFn (Layer.evalState env (finalState interface offset)) =
      ((ProductionKey.key relation ajtai).piCcsExecution
        running fresh proof).outgoingState := by
  have trace := specification
  change Absorb.reference
      (List.ofFn (Layer.evalState env (interface.initialState offset)))
      (Hash.evalList env (outputWords interface offset)) =
    List.ofFn (Layer.evalState env (finalState interface offset)) at trace
  rw [outputWords_eval interface offset env proof.piCcsOutput padEq matrixEq]
    at trace
  calc
    List.ofFn (Layer.evalState env (finalState interface offset)) =
        Absorb.reference
          (List.ofFn (Layer.evalState env (interface.initialState offset)))
          (ProductionKey.fullOutputWords proof.piCcsOutput) := trace.symm
    _ = NightstreamFPrime.Lifecycle.Transcript.absorb
          ((ProductionKey.key relation ajtai).piCcsExecution
            running fresh proof).coins.finalState
          (ProductionKey.fullOutputWords proof.piCcsOutput) := by
      rw [initialEq]
      rfl
    _ = ProductionKey.absorbFullOutput
          ((ProductionKey.key relation ajtai).piCcsExecution
            running fresh proof).coins.finalState proof.piCcsOutput := by
      rfl
    _ = (ProductionKey.key relation ajtai).absorbPiCcsOutput
          ((ProductionKey.key relation ajtai).piCcsExecution
            running fresh proof).coins.finalState proof.piCcsOutput := by
      symm
      exact ProductionKey.key_absorbPiCcsOutput relation ajtai _ _
    _ = ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).outgoingState :=
      ((ProductionKey.key relation ajtai
        ).piCcsExecution_outgoingState_eq_absorbPiCcsOutput
          running fresh proof).symm

theorem key_output_eval_K
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Running
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof (ProductionKey.degreeBound relation))
    (coordinate : PadCoordinate productionShape) :
    ((ProductionKey.key relation ajtai).piCcsCertificate
      running fresh proof).output.padImage coordinate =
        proof.piCcsOutput.padCoordinate
          (runningSourceIndex coordinate.running) coordinate.coefficient := by
  rfl

theorem key_output_eval_A
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Running
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof (ProductionKey.degreeBound relation))
    (coordinate : MatrixCoordinate productionShape) :
    ((ProductionKey.key relation ajtai).piCcsCertificate
      running fresh proof).output.matrixImage coordinate =
        proof.piCcsOutput.matrixCoordinate
          (runningSourceIndex coordinate.running) coordinate.matrix
            coordinate.coefficient := by
  rfl

end NightstreamFPrime.Lifecycle.PiCCS.v1_2.OutputBinding

import NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption
import NightstreamFPrime.Spec.Folding.PiCCS.Accepted

/-!
Paper authority: SuperNeo v1.1, Section 7.3, Steps 3 and 5; the
Fiat–Shamir handoff to `Pi_RLC`.
Obligation: Reuse the complete prover `y′` family as the 17 reduced CE
evaluation families, absorb it in canonical source/Pad/matrix order, and
bind the verifier-owned outgoing transcript state.

Inputs:
- the verifier-derived final SumCheck point and transcript state;
- all 17 separate Pad (`Eval_K`) output families;
- all 17 × 14 CCS-matrix (`Eval_A`) output families.

Outputs:
- zero-copy reduced point, `Eval_K`, and `Eval_A` views;
- the transcript state from which `Pi_RLC` samples all 17 challenges.

Constraint groups:
- C1: one complete length-prefixed output absorption through the generic
  Duplex circuit;
- C2: no final-state or reduced-claim copy rows.

Parent coverage:
- `v1_1.Coverage.output_eval_K`;
- `v1_1.Coverage.output_eval_A`;
- `Key.piCcsExecution.outgoingState`.

The generic Duplex child owns Poseidon2 operations. This leaf owns only the
exact v1.1 output order and the zero-copy protocol handoff.
-/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Poseidon2.Duplex
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources

/-- One shared symbolic copy of the complete paper `y′` family. -/
structure OutputExpr where
  padCoordinate : Fin productionShape.sourceCount →
    Fin productionShape.coefficientCount → KExpr
  matrixCoordinate : Fin productionShape.sourceCount →
    Fin productionShape.matrixCount →
    Fin productionShape.coefficientCount → KExpr

/-- Complete symbolic interface of the output-binding leaf. -/
structure Interface where
  roundPoint : Nat → Fin productionShape.cubeVariables → KExpr
  initialState : Nat → Layer.EState
  output : Nat → OutputExpr

/-- Every reduced CE output reuses the verifier-derived point. -/
def reducedPoint (interface : Interface) (offset : Nat)
    (_source : Fin productionShape.sourceCount) :=
  interface.roundPoint offset

/-- The reduced `Eval_K` view is the same Pad wire family. -/
def reducedEval_K (interface : Interface) (offset : Nat)
    (source : Fin productionShape.sourceCount)
    (coefficient : Fin productionShape.coefficientCount) : KExpr :=
  (interface.output offset).padCoordinate source coefficient

/-- The reduced `Eval_A` view is the same genuine-matrix wire family. -/
def reducedEval_A (interface : Interface) (offset : Nat)
    (source : Fin productionShape.sourceCount)
    (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount) : KExpr :=
  (interface.output offset).matrixCoordinate source matrix coefficient

@[simp] theorem reducedPoint_eq (interface : Interface) (offset : Nat)
    (source : Fin productionShape.sourceCount) :
    reducedPoint interface offset source = interface.roundPoint offset := by
  rfl

@[simp] theorem reducedEval_K_eq (interface : Interface) (offset : Nat)
    (source : Fin productionShape.sourceCount)
    (coefficient : Fin productionShape.coefficientCount) :
    reducedEval_K interface offset source coefficient =
      (interface.output offset).padCoordinate source coefficient := by
  rfl

@[simp] theorem reducedEval_A_eq (interface : Interface) (offset : Nat)
    (source : Fin productionShape.sourceCount)
    (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount) :
    reducedEval_A interface offset source matrix coefficient =
      (interface.output offset).matrixCoordinate source matrix coefficient := by
  rfl

def padWords (output : OutputExpr)
    (source : Fin productionShape.sourceCount) : List Expr :=
  (List.finRange productionShape.coefficientCount).flatMap fun coefficient =>
    StatementAbsorption.serializeKExpr
      (output.padCoordinate source coefficient)

def matrixWords (output : OutputExpr)
    (source : Fin productionShape.sourceCount) : List Expr :=
  (List.finRange productionShape.matrixCount).flatMap fun matrix =>
    (List.finRange productionShape.coefficientCount).flatMap fun coefficient =>
      StatementAbsorption.serializeKExpr
        (output.matrixCoordinate source matrix coefficient)

def sourceWords (output : OutputExpr)
    (source : Fin productionShape.sourceCount) : List Expr :=
  padWords output source ++ matrixWords output source

/-- Canonical `K + k` source order, with Pad coefficients before all genuine
matrix coefficients inside each source. -/
def outputWords (interface : Interface) (offset : Nat) : List Expr :=
  (List.finRange productionShape.sourceCount).flatMap fun source =>
    sourceWords (interface.output offset) source

private theorem flatMap_length_constant
    {Index Value : Type}
    (indices : List Index)
    (values : Index → List Value)
    (count : Nat)
    (each : ∀ index, (values index).length = count) :
    (indices.flatMap values).length = indices.length * count := by
  induction indices with
  | nil => simp
  | cons head tail inductionHypothesis =>
      rw [List.flatMap_cons, List.length_append, each,
        inductionHypothesis]
      simp [Nat.succ_mul, Nat.add_comm]

private theorem padWords_length (output : OutputExpr)
    (source : Fin productionShape.sourceCount) :
    (padWords output source).length = 108 := by
  unfold padWords
  rw [flatMap_length_constant _ _ 2 (fun _ => rfl)]
  norm_num [productionShape, productionProfile,
    Phi81MatrixSource.phi81Shape, ringDegree]

private theorem matrixWords_length (output : OutputExpr)
    (source : Fin productionShape.sourceCount) :
    (matrixWords output source).length = 1512 := by
  have coefficientLength : ∀ matrix,
      ((List.finRange productionShape.coefficientCount).flatMap
        fun coefficient => StatementAbsorption.serializeKExpr
          (output.matrixCoordinate source matrix coefficient)).length = 108 := by
    intro matrix
    rw [flatMap_length_constant _ _ 2 (fun _ => rfl)]
    norm_num [productionShape, productionProfile,
      Phi81MatrixSource.phi81Shape, ringDegree]
  unfold matrixWords
  rw [flatMap_length_constant _ _ 108 coefficientLength]
  norm_num [productionShape, productionProfile,
    Phi81MatrixSource.phi81Shape, ringDegree]

private theorem sourceWords_length (output : OutputExpr)
    (source : Fin productionShape.sourceCount) :
    (sourceWords output source).length = 1620 := by
  rw [sourceWords, List.length_append, padWords_length, matrixWords_length]

/-- The complete output contains 27,540 base-field words: separate Pad and
14-matrix coefficients for all 17 sources. -/
theorem outputWords_length (interface : Interface) (offset : Nat) :
    (outputWords interface offset).length = 27540 := by
  unfold outputWords
  rw [flatMap_length_constant _ _ 1620 (sourceWords_length _)]
  norm_num [productionShape, productionProfile,
    Phi81MatrixSource.phi81Shape, Shape.sourceCount]

/-- One length-prefixed output block is the complete post-SumCheck action. -/
def actions (interface : Interface) (offset : Nat) : List Formal.Action :=
  [StatementAbsorption.absorbBlock (outputWords interface offset)]

def duplexInterface (interface : Interface) : Formal.Owned.Interface where
  initial := interface.initialState
  actions := actions interface

/-- The circuit-owned post-PiCCS transcript state. -/
def finalState (interface : Interface) (offset : Nat) : Layer.EState :=
  Formal.Owned.output (duplexInterface interface) offset

theorem finalState_eq_compile (interface : Interface) (offset : Nat) :
    finalState interface offset =
      (Formal.compile offset (interface.initialState offset)
        (actions interface offset)).output := by
  rfl

abbrev Assumptions (interface : Interface) (offset : Nat) (env : Env) : Prop :=
  Formal.Owned.Assumptions (duplexInterface interface) offset env

/-- Named semantic predicate: the exact Poseidon2 output absorption reaches
the declared post-PiCCS state. -/
abbrev SpecHolds (interface : Interface) (offset : Nat) (env : Env) : Prop :=
  Formal.Owned.SpecHolds (duplexInterface interface) offset env

theorem trace_implies_specHolds (interface : Interface) (offset : Nat)
    (env : Env)
    (trace : Formal.TraceHolds
      (List.ofFn (Layer.evalState env (interface.initialState offset)))
      ((actions interface offset).map (Formal.Action.eval env))
      (List.ofFn (Layer.evalState env (finalState interface offset)))) :
    SpecHolds interface offset env :=
  trace

/-- The sole logical circuit for this leaf. -/
def circuit (interface : Interface) : FormalCircuit :=
  Formal.Owned.circuit (duplexInterface interface)

theorem soundness (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (circuit interface).main offset)) :
    SpecHolds interface offset env :=
  Formal.Owned.soundness (duplexInterface interface) env offset assumptions rows

theorem noAssertions (interface : Interface) (offset : Nat) :
    Formal.Owned.allAssertions (duplexInterface interface) offset = [] := by
  rfl

/-- Honest execution owns the final state because this schedule contains only
one absorb action and therefore no squeeze assertion. -/
theorem build (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (circuit interface).main offset)) ∧
      holdsFlat completed (Circuit.ops (circuit interface).main offset) :=
  Formal.Owned.build_of_no_assertions (duplexInterface interface) env offset
    assumptions (noAssertions interface offset)

theorem completeness (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (circuit interface).main offset)) ∧
      holdsFlat completed (Circuit.ops (circuit interface).main offset) :=
  Formal.Owned.completeness (duplexInterface interface) env offset assumptions
    specification

theorem flatConstraints_varsBelow (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints
      (Circuit.ops (circuit interface).main offset),
      expression.VarsBelow
        (offset + localLength (Circuit.ops (circuit interface).main offset)) :=
  Formal.Owned.flatConstraints_varsBelow (duplexInterface interface) offset env
    assumptions

theorem finalState_varsBelow (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    ∀ lane, (finalState interface offset lane).VarsBelow
      (offset + localLength (Circuit.ops (circuit interface).main offset)) :=
  Formal.Owned.output_varsBelow (duplexInterface interface) offset env
    assumptions

theorem actions_length (interface : Interface) (offset : Nat) :
    (actions interface offset).length = 1 := by
  rfl

/-- Exact symbolic private-variable footprint of the output absorption. -/
def recipeCount (interface : Interface) (offset : Nat) : Nat :=
  Formal.recipeCount (actions interface offset)

private theorem inputChunks_length (input : List Expr) :
    (Hash.inputChunks input).length = (input.length + 3) / 4 := by
  unfold Hash.inputChunks
  rw [List.length_map, List.length_range]
  rfl

private theorem blockExpr_length (words : List Expr) :
    (StatementAbsorption.blockExpr words).length = words.length + 1 := by
  simp [StatementAbsorption.blockExpr]

/-- One 27,541-word length-prefixed block uses 6,886 Poseidon2 chunks and
4,076,512 private recipe variables. -/
theorem recipeCount_eq (interface : Interface) (offset : Nat) :
    recipeCount interface offset = 4076512 := by
  unfold recipeCount actions StatementAbsorption.absorbBlock
  simp only [Formal.recipeCount, List.map_cons, List.map_nil,
    List.sum_cons, List.sum_nil, Nat.add_zero, Formal.Action.recipeCount]
  rw [inputChunks_length, blockExpr_length, outputWords_length]

theorem localLength_eq (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (circuit interface).main offset) = 4076512 := by
  change localLength (Formal.Owned.opsAt (duplexInterface interface) offset) = _
  rw [Formal.Owned.opsAt_localLength]
  unfold Formal.Owned.program
  rw [Formal.compile_recipes_length]
  simpa [duplexInterface, recipeCount] using recipeCount_eq interface offset

/-- One witness operation and no boundary assertion. -/
theorem operations_length (interface : Interface) (offset : Nat) :
    (Circuit.ops (circuit interface).main offset).length = 1 := by
  change (Formal.Owned.opsAt (duplexInterface interface) offset).length = 1
  rw [Formal.Owned.operations_length]
  simp [duplexInterface, actions, StatementAbsorption.absorbBlock,
    Formal.assertionCount, Formal.Action.assertionCount]

/-- One row per recipe and no final-state row. -/
theorem flatConstraints_length (interface : Interface) (offset : Nat) :
    (flatConstraints (Circuit.ops (circuit interface).main offset)).length =
      4076512 := by
  change (flatConstraints
    (Formal.Owned.opsAt (duplexInterface interface) offset)).length = _
  rw [Formal.Owned.flatConstraints_length]
  have recipes : Formal.recipeCount
      ((duplexInterface interface).actions offset) = 4076512 := by
    simpa [duplexInterface, recipeCount] using recipeCount_eq interface offset
  rw [recipes]
  simp [duplexInterface, actions, StatementAbsorption.absorbBlock,
    Formal.assertionCount, Formal.Action.assertionCount]

def valuePadWords
    (output : FullOutputCoordinates.FullOutput K productionShape)
    (source : Fin productionShape.sourceCount) : List F :=
  (List.finRange productionShape.coefficientCount).flatMap fun coefficient =>
    NightstreamFPrime.Lifecycle.serializeK
      (output.padCoordinate source coefficient)

def valueMatrixWords
    (output : FullOutputCoordinates.FullOutput K productionShape)
    (source : Fin productionShape.sourceCount) : List F :=
  (List.finRange productionShape.matrixCount).flatMap fun matrix =>
    (List.finRange productionShape.coefficientCount).flatMap fun coefficient =>
      NightstreamFPrime.Lifecycle.serializeK
        (output.matrixCoordinate source matrix coefficient)

def valueWords
    (output : FullOutputCoordinates.FullOutput K productionShape) : List F :=
  (List.finRange productionShape.sourceCount).flatMap fun source =>
    valuePadWords output source ++ valueMatrixWords output source

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

private theorem padWords_eval (interface : Interface) (offset : Nat)
    (env : Env) (output : FullOutputCoordinates.FullOutput K productionShape)
    (padEq : ∀ source coefficient,
      ((interface.output offset).padCoordinate source coefficient).eval env =
        output.padCoordinate source coefficient)
    (source : Fin productionShape.sourceCount) :
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

private theorem matrixWords_eval (interface : Interface) (offset : Nat)
    (env : Env) (output : FullOutputCoordinates.FullOutput K productionShape)
    (matrixEq : ∀ source matrix coefficient,
      ((interface.output offset).matrixCoordinate source matrix
        coefficient).eval env =
        output.matrixCoordinate source matrix coefficient)
    (source : Fin productionShape.sourceCount) :
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

private theorem sourceWords_eval (interface : Interface) (offset : Nat)
    (env : Env) (output : FullOutputCoordinates.FullOutput K productionShape)
    (padEq : ∀ source coefficient,
      ((interface.output offset).padCoordinate source coefficient).eval env =
        output.padCoordinate source coefficient)
    (matrixEq : ∀ source matrix coefficient,
      ((interface.output offset).matrixCoordinate source matrix
        coefficient).eval env =
        output.matrixCoordinate source matrix coefficient)
    (source : Fin productionShape.sourceCount) :
    (sourceWords (interface.output offset) source).map (Expr.eval env) =
      valuePadWords output source ++ valueMatrixWords output source := by
  rw [sourceWords, List.map_append, padWords_eval interface offset env output
    padEq, matrixWords_eval interface offset env output matrixEq]

/-- Symbolic output serialization evaluates bit-for-bit to the semantic
production output serialization. -/
theorem outputWords_eval (interface : Interface) (offset : Nat) (env : Env)
    (output : FullOutputCoordinates.FullOutput K productionShape)
    (padEq : ∀ source coefficient,
      ((interface.output offset).padCoordinate source coefficient).eval env =
        output.padCoordinate source coefficient)
    (matrixEq : ∀ source matrix coefficient,
      ((interface.output offset).matrixCoordinate source matrix
        coefficient).eval env =
        output.matrixCoordinate source matrix coefficient) :
    Hash.evalList env (outputWords interface offset) = valueWords output := by
  unfold Hash.evalList outputWords valueWords
  apply map_flatMap_congr
  intro source
  exact sourceWords_eval interface offset env output padEq matrixEq source

private theorem reference_block_eq_absorbBlock
    (state : NightstreamFPrime.Lifecycle.Transcript.State) (words : List F) :
    Absorb.reference state (NightstreamFPrime.Lifecycle.block words) =
      NightstreamFPrime.Lifecycle.Transcript.absorbBlock state words := by
  rfl

/-- Circuit coverage of the verifier-owned outgoing state. The prover
supplies only `y′`; the final transcript state is recomputed. -/
theorem spec_implies_keyOutgoingState
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
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
      (Hash.evalList env
        (StatementAbsorption.blockExpr (outputWords interface offset))) =
    List.ofFn (Layer.evalState env (finalState interface offset)) at trace
  have wordsEq := outputWords_eval interface offset env proof.piCcsOutput
    padEq matrixEq
  have blockEq : Hash.evalList env
      (StatementAbsorption.blockExpr (outputWords interface offset)) =
      NightstreamFPrime.Lifecycle.block (valueWords proof.piCcsOutput) := by
    unfold StatementAbsorption.blockExpr NightstreamFPrime.Lifecycle.block
    change NightstreamFPrime.Lifecycle.natWord
        (outputWords interface offset).length ::
          Hash.evalList env (outputWords interface offset) =
      NightstreamFPrime.Lifecycle.natWord
        (valueWords proof.piCcsOutput).length ::
          valueWords proof.piCcsOutput
    have lengthEq : (outputWords interface offset).length =
        (valueWords proof.piCcsOutput).length := by
      calc
        (outputWords interface offset).length =
            (Hash.evalList env (outputWords interface offset)).length := by
          simp [Hash.evalList]
        _ = (valueWords proof.piCcsOutput).length :=
          congrArg List.length wordsEq
    rw [wordsEq, lengthEq]
  rw [blockEq] at trace
  calc
    List.ofFn (Layer.evalState env (finalState interface offset)) =
        Absorb.reference
          (List.ofFn (Layer.evalState env (interface.initialState offset)))
          (NightstreamFPrime.Lifecycle.block
            (valueWords proof.piCcsOutput)) := trace.symm
    _ = NightstreamFPrime.Lifecycle.Transcript.absorbBlock
          (List.ofFn (Layer.evalState env (interface.initialState offset)))
          (valueWords proof.piCcsOutput) :=
      reference_block_eq_absorbBlock _ _
    _ = NightstreamFPrime.Lifecycle.Transcript.absorbBlock
          ((ProductionKey.key relation ajtai).piCcsExecution
            running fresh proof).coins.finalState
          (valueWords proof.piCcsOutput) := by rw [initialEq]
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

/-- Exact output-Pad conjunct used by the canonical v1.1 coverage map. -/
theorem key_output_eval_K
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
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

/-- Exact output-matrix conjunct used by the canonical v1.1 coverage map. -/
theorem key_output_eval_A
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
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

end NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding

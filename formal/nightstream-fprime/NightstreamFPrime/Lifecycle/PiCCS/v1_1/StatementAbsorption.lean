import NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal
import NightstreamFPrime.Lifecycle.ProductionKey
import NightstreamFPrime.Spec.Folding.PiCCS.Statement

/-!
Paper authority: SuperNeo v1.1, Section 7.3, public input and Step 1;
Fiat–Shamir transform of the public-coin verifier transcript.
Obligation: Absorb the pilot-bound prior-state digest and the fresh public
claim before deriving `α`, `γ`, or any SumCheck challenge.

Inputs:
- the digest-only PiCCS domain tag;
- the four digest lanes projected from the pilot-bound fresh public input;
- one fresh commitment and public input.

Outputs:
- the Poseidon2 state after the complete statement absorption.

Constraint groups:
- C1: four ordered absorb actions through the generic Duplex circuit;
- C2: eight final-state equality constraints.

Parent coverage:
- `v1_1.Coverage.transcript` committed-statement prefix;
- pilot-to-PiCCS prior-digest wiring;
- the fresh statement.

This leaf contains no squeeze and accepts no witness-supplied challenge. The
complete running statement remains available to the other PiCCS leaves, but
this leaf does not absorb it again. The generic Duplex child owns all
Poseidon2 operations; this leaf owns only typed serialization and wiring.
-/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Poseidon2.Duplex
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive

/-- Symbolic form of one complete v1.1 CE evaluation family. -/
structure EvaluationExpr where
  eval_K : Fin productionShape.coefficientCount → KExpr
  eval_A : Fin productionShape.matrixCount →
    Fin productionShape.coefficientCount → KExpr

/-- Symbolic running public claims. -/
structure RunningExpr (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) where
  point : Fin productionShape.cubeVariables → KExpr
  commitment : Fin productionShape.runningCount →
    Fin productionProfile.commitmentWidth → Fin ringDegree → Expr
  publicInput : Fin productionShape.runningCount →
    Fin (FullShape logicalWidth publicFits).publicWidth → Expr
  evaluation : Fin productionShape.runningCount → EvaluationExpr

/-- Symbolic fresh public claims. -/
structure FreshExpr (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) where
  commitment : Fin productionShape.freshCount →
    Fin productionProfile.commitmentWidth → Fin ringDegree → Expr
  publicInput : Fin productionShape.freshCount →
    Fin (FullShape logicalWidth publicFits).publicWidth → Expr

/-- External symbolic inputs of the statement-absorption leaf. The final
state is compiler output and is not supplied by the caller. -/
structure Interface (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) where
  running : Nat → RunningExpr logicalWidth publicFits
  fresh : Nat → FreshExpr logicalWidth publicFits

def evalPoint (point : Fin productionShape.cubeVariables → KExpr)
    (env : Env) : CubePoint K productionShape.cubeVariables where
  coordinates := List.ofFn fun coordinate => (point coordinate).eval env
  dimension := by simp

def evalEvaluation (evaluation : EvaluationExpr) (env : Env) :
    StrongReduction.EvaluationFamily K productionShape where
  pad := fun coefficient => (evaluation.eval_K coefficient).eval env
  matrix := fun matrix coefficient =>
    (evaluation.eval_A matrix coefficient).eval env

def evalRunning {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : RunningExpr logicalWidth publicFits) (env : Env) :
    NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive.Running K
      PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput
        (logicalWidth := logicalWidth) (publicFits := publicFits))
      productionShape where
  point := evalPoint running.point env
  commitments := fun source row coefficient =>
    (running.commitment source row coefficient).eval env
  publicInputs := fun source column =>
    (running.publicInput source column).eval env
  evaluations := fun source => evalEvaluation (running.evaluation source) env

def evalFresh {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (fresh : FreshExpr logicalWidth publicFits) (env : Env) :
    NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive.Fresh
      PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput
        (logicalWidth := logicalWidth) (publicFits := publicFits))
      productionShape where
  commitments := fun source row coefficient =>
    (fresh.commitment source row coefficient).eval env
  publicInputs := fun source column =>
    (fresh.publicInput source column).eval env

def serializeKExpr (value : KExpr) : List Expr := [value.c0, value.c1]

def serializePointExpr
    (point : Fin productionShape.cubeVariables → KExpr) : List Expr :=
  (List.finRange productionShape.cubeVariables).flatMap fun coordinate =>
    serializeKExpr (point coordinate)

def serializeCommitmentExpr
    (commitment : Fin productionProfile.commitmentWidth →
      Fin ringDegree → Expr) : List Expr :=
  (List.finRange productionProfile.commitmentWidth).flatMap fun row =>
    (List.finRange ringDegree).map fun coefficient =>
      commitment row coefficient

def serializePublicInputExpr {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (input : Fin (FullShape logicalWidth publicFits).publicWidth → Expr) :
    List Expr :=
  (List.finRange (FullShape logicalWidth publicFits).publicWidth).map input

/-- Serialize `Eval_K` first and all genuine `Eval_A` matrices second. -/
def serializeEvaluationExpr (evaluation : EvaluationExpr) : List Expr :=
  ((List.finRange productionShape.coefficientCount).flatMap fun coefficient =>
      serializeKExpr (evaluation.eval_K coefficient)) ++
    (List.finRange productionShape.matrixCount).flatMap fun matrix =>
      (List.finRange productionShape.coefficientCount).flatMap
        fun coefficient => serializeKExpr (evaluation.eval_A matrix coefficient)

private theorem serializeKExpr_length (value : KExpr) :
    (serializeKExpr value).length = 2 := by
  rfl

private theorem serializePointExpr_length
    (point : Fin productionShape.cubeVariables → KExpr) :
    (serializePointExpr point).length = 50 := by
  simp [serializePointExpr, serializeKExpr_length, productionShape,
    Phi81MatrixSource.phi81Shape, cubeVariables]

private theorem serializeCommitmentExpr_length
    (commitment : Fin productionProfile.commitmentWidth →
      Fin ringDegree → Expr) :
    (serializeCommitmentExpr commitment).length = 972 := by
  simp [serializeCommitmentExpr, productionProfile, ringDegree]

private theorem serializePublicInputExpr_length {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (input : Fin (FullShape logicalWidth publicFits).publicWidth → Expr) :
    (serializePublicInputExpr input).length = 54 := by
  simp [serializePublicInputExpr, fullShape,
    Phi81Relation.Shape.publicWidth, publicRingColumns, ringDegree]

private theorem serializeEvaluationExpr_length (evaluation : EvaluationExpr) :
    (serializeEvaluationExpr evaluation).length = 1620 := by
  simp [serializeEvaluationExpr, serializeKExpr_length, productionShape,
    productionProfile, Phi81MatrixSource.phi81Shape, ringDegree]

def constantWords (words : List F) : List Expr := words.map Expr.const

def blockExpr (words : List Expr) : List Expr :=
  Expr.const (NightstreamFPrime.Lifecycle.natWord words.length) :: words

private theorem constantWords_length (words : List F) :
    (constantWords words).length = words.length := by
  simp [constantWords]

private theorem blockExpr_length (words : List Expr) :
    (blockExpr words).length = words.length + 1 := by
  simp [blockExpr]

def absorbBlock (words : List Expr) : Formal.Action :=
  .absorb (blockExpr words)

/-- The four prior-digest lanes are definitionally projected from the fresh
public input that the pilot binds to `[1, digest, 0…]`. -/
def priorDigestExpr {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) : List Expr :=
  List.ofFn fun lane : Fin 4 =>
    (interface.fresh offset).publicInput ⟨0, by decide⟩
      (ProductionKey.priorDigestIndex lane)

/-- Symbolic form of the digest-only `ProductionKey.publicInputBlocks`. -/
def publicInputBlocks {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) : List (List Expr) :=
  let fresh := interface.fresh offset
  [priorDigestExpr interface offset] ++
  (List.finRange productionShape.freshCount).flatMap fun index =>
    [serializeCommitmentExpr (fresh.commitment index),
      serializePublicInputExpr (fresh.publicInput index)]

/-- Key-owned public NIFS input prefix. -/
def publicInputActions {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) : List Formal.Action :=
  [.absorb (constantWords
      NightstreamFPrime.Lifecycle.Transcript.piCcsDigestDomainTag)] ++
    (publicInputBlocks interface offset).map absorbBlock

private theorem publicInputActions_eq {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    publicInputActions interface offset =
      let fresh := interface.fresh offset
      [.absorb (constantWords
          NightstreamFPrime.Lifecycle.Transcript.piCcsDigestDomainTag),
        absorbBlock (priorDigestExpr interface offset)] ++
      (List.finRange productionShape.freshCount).flatMap fun index =>
        [absorbBlock (serializeCommitmentExpr (fresh.commitment index)),
          absorbBlock (serializePublicInputExpr (fresh.publicInput index))] := by
  unfold publicInputActions publicInputBlocks
  dsimp only
  simp [List.map_flatMap]

/-- Verifier-owned claim words in exact v1.1 `Eval_K`, then `Eval_A`, order. -/
def verifierClaimWords {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) : List Expr :=
  let running := interface.running offset
  ((canonicalPadCoordinates productionShape).flatMap fun coordinate =>
      serializeKExpr
        ((running.evaluation coordinate.running).eval_K
          coordinate.coefficient)) ++
    (canonicalMatrixCoordinates productionShape).flatMap fun coordinate =>
      serializeKExpr ((running.evaluation coordinate.running).eval_A
        coordinate.matrix coordinate.coefficient)

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

private theorem verifierClaimWords_length {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (verifierClaimWords interface offset).length = 25920 := by
  let running := interface.running offset
  have padLength :
      ((canonicalPadCoordinates productionShape).flatMap fun coordinate =>
        serializeKExpr
          ((running.evaluation coordinate.running).eval_K
            coordinate.coefficient)).length =
        (canonicalPadCoordinates productionShape).length * 2 := by
    apply flatMap_length_constant
    intro coordinate
    exact serializeKExpr_length _
  have matrixLength :
      ((canonicalMatrixCoordinates productionShape).flatMap fun coordinate =>
        serializeKExpr
          ((running.evaluation coordinate.running).eval_A coordinate.matrix
            coordinate.coefficient)).length =
        (canonicalMatrixCoordinates productionShape).length * 2 := by
    apply flatMap_length_constant
    intro coordinate
    exact serializeKExpr_length _
  unfold verifierClaimWords
  dsimp only
  rw [List.length_append, padLength, matrixLength,
    canonicalPadCoordinates_length, canonicalMatrixCoordinates_length]
  norm_num [productionShape, productionProfile,
    Phi81MatrixSource.phi81Shape, Shape.padEvaluationCount,
    Shape.matrixEvaluationCount, ringDegree]

/-- The two verifier-owned blocks: prior point, then `Eval_K ++ Eval_A`. -/
def verifierInputBlocks {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) : List (List Expr) :=
  [serializePointExpr (interface.running offset).point,
    verifierClaimWords interface offset]

/-- The public verifier input is absorbed after the key-owned public prefix. -/
def verifierInputActions {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) : List Formal.Action :=
  (verifierInputBlocks interface offset).map absorbBlock

private theorem verifierInputActions_eq {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    verifierInputActions interface offset =
      [absorbBlock (serializePointExpr (interface.running offset).point),
        absorbBlock (verifierClaimWords interface offset)] := by
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

@[simp] private theorem serializeKExpr_eval (env : Env) (value : KExpr) :
    (serializeKExpr value).map (Expr.eval env) =
      NightstreamFPrime.Lifecycle.serializeK (value.eval env) := by
  rfl

private theorem serializePointExpr_eval
    (point : Fin productionShape.cubeVariables → KExpr) (env : Env) :
    Hash.evalList env (serializePointExpr point) =
      NightstreamFPrime.Lifecycle.serializePoint (evalPoint point env) := by
  unfold Hash.evalList serializePointExpr
    NightstreamFPrime.Lifecycle.serializePoint
  have coordinatesEq : (evalPoint point env).coordinates =
      (List.finRange productionShape.cubeVariables).map fun coordinate =>
        (point coordinate).eval env := by
    exact List.ofFn_eq_map
  rw [coordinatesEq, List.flatMap_map]
  apply map_flatMap_congr
  intro coordinate
  exact serializeKExpr_eval env (point coordinate)

private theorem serializeCommitmentExpr_eval
    (commitment : Fin productionProfile.commitmentWidth →
      Fin ringDegree → Expr) (env : Env) :
    Hash.evalList env (serializeCommitmentExpr commitment) =
      NightstreamFPrime.Lifecycle.serializeCommitment
        (fun row coefficient => (commitment row coefficient).eval env) := by
  unfold Hash.evalList serializeCommitmentExpr
    NightstreamFPrime.Lifecycle.serializeCommitment
  apply map_flatMap_congr
  intro row
  simp [NightstreamFPrime.Lifecycle.serializeRingF, List.map_map,
    Function.comp_def]

private theorem serializePublicInputExpr_eval
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (input : Fin (FullShape logicalWidth publicFits).publicWidth → Expr)
    (env : Env) :
    Hash.evalList env (serializePublicInputExpr input) =
      NightstreamFPrime.Lifecycle.serializePublicInput
        (publicFits := publicFits) (fun column => (input column).eval env) := by
  unfold Hash.evalList serializePublicInputExpr
    NightstreamFPrime.Lifecycle.serializePublicInput
  simp [List.map_map, Function.comp_def]

private theorem serializeEvaluationExpr_eval
    (evaluation : EvaluationExpr) (env : Env) :
    Hash.evalList env (serializeEvaluationExpr evaluation) =
      NightstreamFPrime.Lifecycle.serializeEvaluations
        (evalEvaluation evaluation env) := by
  unfold Hash.evalList serializeEvaluationExpr
    NightstreamFPrime.Lifecycle.serializeEvaluations evalEvaluation
  rw [List.map_append]
  apply congrArg₂ List.append
  · apply map_flatMap_congr
    intro coefficient
    exact serializeKExpr_eval env (evaluation.eval_K coefficient)
  · apply map_flatMap_congr
    intro matrix
    apply map_flatMap_congr
    intro coefficient
    exact serializeKExpr_eval env (evaluation.eval_A matrix coefficient)

private theorem runningGroup_eval {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : RunningExpr logicalWidth publicFits) (env : Env)
    (index : Fin productionShape.runningCount) :
    [serializeCommitmentExpr (running.commitment index),
        serializePublicInputExpr (running.publicInput index),
        serializeEvaluationExpr (running.evaluation index)].map
        (Hash.evalList env) =
      [NightstreamFPrime.Lifecycle.serializeCommitment
          ((evalRunning running env).commitments index),
        NightstreamFPrime.Lifecycle.serializePublicInput
          (publicFits := publicFits) ((evalRunning running env).publicInputs index),
        NightstreamFPrime.Lifecycle.serializeEvaluations
          ((evalRunning running env).evaluations index)] := by
  simp only [List.map_cons, List.map_nil]
  rw [serializeCommitmentExpr_eval, serializePublicInputExpr_eval,
    serializeEvaluationExpr_eval]
  rfl

private theorem freshGroup_eval {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (fresh : FreshExpr logicalWidth publicFits) (env : Env)
    (index : Fin productionShape.freshCount) :
    [serializeCommitmentExpr (fresh.commitment index),
        serializePublicInputExpr (fresh.publicInput index)].map
        (Hash.evalList env) =
      [NightstreamFPrime.Lifecycle.serializeCommitment
          ((evalFresh fresh env).commitments index),
        NightstreamFPrime.Lifecycle.serializePublicInput
          (publicFits := publicFits) ((evalFresh fresh env).publicInputs index)] := by
  simp only [List.map_cons, List.map_nil]
  rw [serializeCommitmentExpr_eval, serializePublicInputExpr_eval]
  rfl

private theorem priorDigestExpr_eval {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) (env : Env) :
    Hash.evalList env (priorDigestExpr interface offset) =
      ProductionKey.priorDigest
        (evalFresh (interface.fresh offset) env) := by
  unfold Hash.evalList priorDigestExpr ProductionKey.priorDigest
  rw [List.map_ofFn]
  rfl

/-- The symbolic public block list evaluates exactly to the production key's
canonical semantic block list. -/
theorem publicInputBlocks_eval {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) (env : Env) :
    (publicInputBlocks interface offset).map (Hash.evalList env) =
      ProductionKey.publicInputBlocks
        (evalRunning (interface.running offset) env)
        (evalFresh (interface.fresh offset) env) := by
  unfold publicInputBlocks ProductionKey.publicInputBlocks
  dsimp only
  simp only [List.map_append]
  apply congrArg₂ List.append
  · simp only [List.map_cons, List.map_nil]
    rw [priorDigestExpr_eval]
  · apply map_flatMap_congr
    intro index
    exact freshGroup_eval (interface.fresh offset) env index

private theorem verifierClaimWords_eval {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) (env : Env) :
    Hash.evalList env (verifierClaimWords interface offset) =
      ((canonicalPadCoordinates productionShape).flatMap fun coordinate =>
        NightstreamFPrime.Lifecycle.serializeK
          ((evalEvaluation
            ((interface.running offset).evaluation coordinate.running) env
              ).pad coordinate.coefficient)) ++
      ((canonicalMatrixCoordinates productionShape).flatMap fun coordinate =>
        NightstreamFPrime.Lifecycle.serializeK
          ((evalEvaluation
            ((interface.running offset).evaluation coordinate.running) env
              ).matrix coordinate.matrix coordinate.coefficient)) := by
  unfold Hash.evalList verifierClaimWords
  dsimp only
  rw [List.map_append]
  apply congrArg₂ List.append
  · apply map_flatMap_congr
    intro coordinate
    exact serializeKExpr_eval env
      (((interface.running offset).evaluation coordinate.running).eval_K
        coordinate.coefficient)
  · apply map_flatMap_congr
    intro coordinate
    exact serializeKExpr_eval env
      (((interface.running offset).evaluation coordinate.running).eval_A
        coordinate.matrix coordinate.coefficient)

/-- The symbolic verifier blocks evaluate exactly to the production key's
canonical v1.1 verifier input blocks. -/
theorem verifierInputBlocks_eval
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Interface logicalWidth publicFits) (offset : Nat) (env : Env) :
    (verifierInputBlocks interface offset).map (Hash.evalList env) =
      let running := evalRunning (interface.running offset) env
      let fresh := evalFresh (interface.fresh offset) env
      let key := ProductionKey.key relation ajtai
      NightstreamFPrime.Lifecycle.Transcript.verifierInputBlocks
        ((key.statement running fresh).verifierInput key.lift) := by
  dsimp only
  unfold verifierInputBlocks
    NightstreamFPrime.Lifecycle.Transcript.verifierInputBlocks
  simp only [List.map_cons, List.map_nil]
  rw [serializePointExpr_eval, verifierClaimWords_eval]
  simp [StrongReduction.Statement.verifierInput, Key.statement, evalRunning]

def absorbedBlocks {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    List (List Expr) :=
  publicInputBlocks interface offset

theorem absorbedBlocks_eval
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) (env : Env) :
    (absorbedBlocks interface offset).map (Hash.evalList env) =
      let running := evalRunning (interface.running offset) env
      let fresh := evalFresh (interface.fresh offset) env
      ProductionKey.publicInputBlocks running fresh := by
  dsimp only
  exact publicInputBlocks_eval interface offset env

private theorem constantWords_eval (env : Env) (words : List F) :
    Hash.evalList env (constantWords words) = words := by
  simp [Hash.evalList, constantWords, Function.comp_def]

private theorem blockExpr_eval (env : Env) (words : List Expr) :
    Hash.evalList env (blockExpr words) =
      NightstreamFPrime.Lifecycle.block (Hash.evalList env words) := by
  simp [Hash.evalList, blockExpr, NightstreamFPrime.Lifecycle.block]

private theorem absorbBlockActions_eval (env : Env)
    (blocks : List (List Expr)) :
    (blocks.map absorbBlock).map (Formal.Action.eval env) =
      blocks.map fun words => Formal.ValueAction.absorb
        (NightstreamFPrime.Lifecycle.block (Hash.evalList env words)) := by
  induction blocks with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons]
      apply congrArg₂ List.cons
      · change Formal.ValueAction.absorb
          (Hash.evalList env (blockExpr head)) =
            Formal.ValueAction.absorb
              (NightstreamFPrime.Lifecycle.block (Hash.evalList env head))
        exact congrArg Formal.ValueAction.absorb (blockExpr_eval env head)
      · exact inductionHypothesis

private theorem reference_eq_absorb
    (state : NightstreamFPrime.Lifecycle.Transcript.State) (words : List F) :
    Absorb.reference state words =
      NightstreamFPrime.Lifecycle.Transcript.absorb state words := by
  rfl

private theorem reference_block_eq_absorbBlock
    (state : NightstreamFPrime.Lifecycle.Transcript.State) (words : List F) :
    Absorb.reference state (NightstreamFPrime.Lifecycle.block words) =
      NightstreamFPrime.Lifecycle.Transcript.absorbBlock state words := by
  rfl

private theorem traceHolds_absorbBlocks_iff
    (state final : NightstreamFPrime.Lifecycle.Transcript.State)
    (blocks : List (List F)) :
    Formal.TraceHolds state
        (blocks.map fun words => Formal.ValueAction.absorb
          (NightstreamFPrime.Lifecycle.block words)) final ↔
      NightstreamFPrime.Lifecycle.Transcript.absorbBlocks state blocks =
        final := by
  induction blocks generalizing state with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      change Formal.TraceHolds
          (Absorb.reference state (NightstreamFPrime.Lifecycle.block head))
          (tail.map fun words => Formal.ValueAction.absorb
            (NightstreamFPrime.Lifecycle.block words)) final ↔ _
      rw [reference_block_eq_absorbBlock, inductionHypothesis]
      rfl

private theorem inputChunks_length (input : List Expr) :
    (Hash.inputChunks input).length = (input.length + 3) / 4 := by
  unfold Hash.inputChunks
  rw [List.length_map, List.length_range]
  rfl

private theorem absorb_recipeCount (input : List Expr) :
    Formal.Action.recipeCount (.absorb input) =
      ((input.length + 3) / 4) * 592 := by
  change (Hash.inputChunks input).length * 592 = _
  rw [inputChunks_length]

@[simp] private theorem domain_recipeCount :
    Formal.Action.recipeCount
        (.absorb (constantWords
          NightstreamFPrime.Lifecycle.Transcript.piCcsDigestDomainTag)) =
      6512 := by
  rw [absorb_recipeCount, constantWords_length]
  rw [NightstreamFPrime.Lifecycle.Transcript.piCcsDigestDomainTag_length]

@[simp] private theorem priorDigest_recipeCount {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    Formal.Action.recipeCount
        (absorbBlock (priorDigestExpr interface offset)) = 1184 := by
  unfold absorbBlock priorDigestExpr
  rw [absorb_recipeCount, blockExpr_length]
  simp

@[simp] private theorem point_recipeCount
    (point : Fin productionShape.cubeVariables → KExpr) :
    Formal.Action.recipeCount
        (absorbBlock (serializePointExpr point)) = 7696 := by
  unfold absorbBlock
  rw [absorb_recipeCount, blockExpr_length, serializePointExpr_length]

@[simp] private theorem commitment_recipeCount
    (commitment : Fin productionProfile.commitmentWidth →
      Fin ringDegree → Expr) :
    Formal.Action.recipeCount
        (absorbBlock (serializeCommitmentExpr commitment)) = 144448 := by
  unfold absorbBlock
  rw [absorb_recipeCount, blockExpr_length, serializeCommitmentExpr_length]

@[simp] private theorem publicInput_recipeCount {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (input : Fin (FullShape logicalWidth publicFits).publicWidth → Expr) :
    Formal.Action.recipeCount
        (absorbBlock (serializePublicInputExpr input)) = 8288 := by
  unfold absorbBlock
  rw [absorb_recipeCount, blockExpr_length, serializePublicInputExpr_length]

@[simp] private theorem evaluation_recipeCount (evaluation : EvaluationExpr) :
    Formal.Action.recipeCount
        (absorbBlock (serializeEvaluationExpr evaluation)) = 240352 := by
  unfold absorbBlock
  rw [absorb_recipeCount, blockExpr_length, serializeEvaluationExpr_length]

@[simp] private theorem verifierClaims_recipeCount {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    Formal.Action.recipeCount
        (absorbBlock (verifierClaimWords interface offset)) = 3836752 := by
  unfold absorbBlock
  rw [absorb_recipeCount, blockExpr_length,
    verifierClaimWords_length]

private theorem runningGroup_recipeCount {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : RunningExpr logicalWidth publicFits)
    (index : Fin productionShape.runningCount) :
    Formal.recipeCount
      [absorbBlock (serializeCommitmentExpr (running.commitment index)),
        absorbBlock (serializePublicInputExpr (running.publicInput index)),
        absorbBlock (serializeEvaluationExpr (running.evaluation index))] =
      393088 := by
  simp [Formal.recipeCount]

private theorem freshGroup_recipeCount {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (fresh : FreshExpr logicalWidth publicFits)
    (index : Fin productionShape.freshCount) :
    Formal.recipeCount
      [absorbBlock (serializeCommitmentExpr (fresh.commitment index)),
        absorbBlock (serializePublicInputExpr (fresh.publicInput index))] =
      152736 := by
  simp [Formal.recipeCount]

private theorem publicInputActions_recipeCount {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    Formal.recipeCount (publicInputActions interface offset) = 160432 := by
  let fresh := interface.fresh offset
  have freshCost : Formal.recipeCount
      ((List.finRange productionShape.freshCount).flatMap fun index =>
        [absorbBlock (serializeCommitmentExpr (fresh.commitment index)),
          absorbBlock (serializePublicInputExpr (fresh.publicInput index))]) =
      productionShape.freshCount * 152736 := by
    apply Formal.recipeCount_flatMap_constant
    intro index _
    exact freshGroup_recipeCount fresh index
  rw [publicInputActions_eq]
  dsimp only
  simp only [Formal.recipeCount_append]
  rw [freshCost]
  simp [Formal.recipeCount, productionShape, productionProfile,
    Phi81MatrixSource.phi81Shape]

private theorem verifierInputActions_recipeCount {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    Formal.recipeCount (verifierInputActions interface offset) = 3844448 := by
  rw [verifierInputActions_eq]
  simp [Formal.recipeCount]

def actions {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) : List Formal.Action :=
  publicInputActions interface offset

private theorem actions_eq {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    actions interface offset =
      [.absorb (constantWords
        NightstreamFPrime.Lifecycle.Transcript.piCcsDigestDomainTag)] ++
        (absorbedBlocks interface offset).map absorbBlock := by
  rfl

private theorem absorbBlocks_assertionCount (blocks : List (List Expr)) :
    Formal.assertionCount (blocks.map absorbBlock) = 0 := by
  induction blocks with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      unfold Formal.assertionCount
      simp only [List.map_cons, List.sum_cons, absorbBlock,
        Formal.Action.assertionCount, Nat.zero_add]
      simpa [Formal.assertionCount, List.map_map, Function.comp_def] using
        inductionHypothesis

theorem assertionCount_eq {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    Formal.assertionCount (actions interface offset) = 0 := by
  rw [actions_eq, Formal.assertionCount_append,
    absorbBlocks_assertionCount]
  rfl

/-- The one causal program that owns the complete statement transcript. -/
def program {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    Formal.Program :=
  Formal.compile offset Hash.zeroE (actions interface offset)

/-- The statement transcript state is the program output. It is never a
caller-supplied witness value. -/
def finalState {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    Layer.EState :=
  (program interface offset).output

def duplexInterface {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) : Formal.Interface where
  initial := fun _ => Hash.zeroE
  actions := actions interface
  final := finalState interface

/-- Only external absorb inputs must precede the call. The derived final
state lives in this leaf's own recipe interval. -/
def Assumptions {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) (_env : Env) : Prop :=
  Formal.ActionsBelow offset (actions interface offset)

/-- Named semantic predicate: the exact Poseidon2 trace of all 54 statement
absorptions reaches the declared output state. -/
def SpecHolds {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) : Prop :=
  Formal.TraceHolds
    (List.ofFn (Layer.evalState env Hash.zeroE))
    ((actions interface offset).map (Formal.Action.eval env))
    (List.ofFn (Layer.evalState env (finalState interface offset)))

/-- The leaf emits only its causal witness batch. Since every action is an
absorption, the compiler has no non-recipe assertion. -/
def opsAt {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) : List Op :=
  [Op.witness (WitnessBatch.arithmetic offset
    (program interface offset).recipes)]

def main {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) : Circuit Unit :=
  fun offset =>
    ((), offset + (program interface offset).recipes.length,
      opsAt interface offset)

@[simp] theorem main_ops {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    Circuit.ops (main interface) offset = opsAt interface offset := by
  rfl

@[simp] theorem opsAt_localLength {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    localLength (opsAt interface offset) =
      (program interface offset).recipes.length := by
  simp [opsAt, localLength, Op.localLength]

@[simp] theorem flatConstraints_opsAt {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    flatConstraints (opsAt interface offset) =
      recipeConstraints offset (program interface offset).recipes := by
  simp [opsAt, flatConstraints, Op.flatConstraints]

private theorem program_assertions_eq_nil {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (program interface offset).assertions = [] := by
  apply List.eq_nil_of_length_eq_zero
  change (Formal.compile offset Hash.zeroE
    (actions interface offset)).assertions.length = 0
  rw [Formal.compile_assertions_length, assertionCount_eq]

/-- Honest execution constructs the complete absorb-only statement trace.
No semantic premise is necessary because this schedule has no assertions. -/
theorem build {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (main interface) offset)) ∧
      holdsFlat completed (Circuit.ops (main interface) offset) := by
  let compiled := program interface offset
  let completed := executeRecipes env offset compiled.recipes
  have causal : RecipesCausal offset compiled.recipes := by
    apply Formal.compile_causal offset Hash.zeroE
      (actions interface offset)
    · intro lane
      trivial
    · exact assumptions
  have recipeRows : ConstraintsHold completed
      (recipeConstraints offset compiled.recipes) :=
    executeRecipes_holds_recipeConstraints env offset compiled.recipes causal
  refine ⟨completed, ?_, ?_⟩
  · simpa only [main_ops, opsAt, localLength, List.map_singleton,
      List.sum_singleton, Op.localLength,
      WitnessBatch.arithmetic_outputLength] using
      executeRecipes_agreesOutside env offset compiled.recipes
  · change ConstraintsHold completed (flatConstraints (opsAt interface offset))
    rw [flatConstraints_opsAt]
    exact recipeRows

/-- The sole logical circuit for this leaf. -/
def circuit {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) : FormalCircuit where
  main := main interface
  assumptions := Assumptions interface
  spec := SpecHolds interface
  soundness := by
    intro env offset _assumptions rows
    let compiled := program interface offset
    have recipeRows : ConstraintsHold env
        (recipeConstraints offset compiled.recipes) := by
      exact rows (Op.witness (WitnessBatch.arithmetic offset compiled.recipes)) (by
        change Op.witness (WitnessBatch.arithmetic offset compiled.recipes) ∈
          opsAt interface offset
        simp [opsAt, compiled])
    have assertionRows : ConstraintsHold env compiled.assertions := by
      rw [show compiled.assertions = [] by
        simpa [compiled] using program_assertions_eq_nil interface offset]
      intro expression member
      cases member
    exact Formal.compile_sound env offset Hash.zeroE
      (actions interface offset) recipeRows assertionRows
  completeness := by
    intro env offset assumptions _specification
    exact build interface env offset assumptions

@[simp] theorem circuit_ops {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    Circuit.ops (circuit interface).main offset = opsAt interface offset := by
  rfl

theorem soundness {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (circuit interface).main offset)) :
    SpecHolds interface offset env :=
  (circuit interface).soundness env offset assumptions rows

theorem completeness {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (circuit interface).main offset)) ∧
      holdsFlat completed (Circuit.ops (circuit interface).main offset) :=
  (circuit interface).completeness env offset assumptions specification

theorem specHolds_of_agree_below {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat)
    (before after : Env) (assumptions : Assumptions interface offset before)
    (agrees : ∀ index,
      index < offset + (program interface offset).recipes.length →
        after index = before index)
    (specification : SpecHolds interface offset before) :
    SpecHolds interface offset after := by
  have initialEval :
      List.ofFn (Layer.evalState after Hash.zeroE) =
        List.ofFn (Layer.evalState before Hash.zeroE) := by
    rfl
  have actionsEval :
      (actions interface offset).map (Formal.Action.eval after) =
        (actions interface offset).map (Formal.Action.eval before) := by
    apply List.map_congr_left
    intro action member
    exact Formal.action_eval_preserved before after offset action
      (assumptions action member) (fun index below => agrees index (by omega))
  have outputBelow := (Formal.compile_scope offset Hash.zeroE
    (actions interface offset) (by intro lane; trivial) assumptions).1
  have finalEval :
      List.ofFn (Layer.evalState after (finalState interface offset)) =
        List.ofFn (Layer.evalState before (finalState interface offset)) := by
    apply congrArg List.ofFn
    funext lane
    exact (finalState interface offset lane).eval_eq_of_agree_below
      (offset + (program interface offset).recipes.length) after before
      (by simpa [finalState, program] using outputBelow lane) agrees
  unfold SpecHolds at specification ⊢
  rw [initialEval, actionsEval, finalEval]
  exact specification

theorem flatConstraints_varsBelow {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints
      (Circuit.ops (circuit interface).main offset),
      expression.VarsBelow
        (offset + localLength (Circuit.ops (circuit interface).main offset)) := by
  have causal : RecipesCausal offset (program interface offset).recipes := by
    apply Formal.compile_causal offset Hash.zeroE (actions interface offset)
    · intro lane
      trivial
    · exact assumptions
  have scope := recipeConstraints_varsBelow_of_causal offset
    (program interface offset).recipes causal
  rw [circuit_ops, flatConstraints_opsAt, opsAt_localLength]
  exact scope

private theorem publicInputBlocks_length {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (publicInputBlocks interface offset).length = 3 := by
  unfold publicInputBlocks
  dsimp only
  simp [productionShape, productionProfile,
    Phi81MatrixSource.phi81Shape]

/-- There are four independently auditable digest-only absorb actions. -/
theorem actions_length {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (actions interface offset).length = 4 := by
  simp [actions, publicInputActions, publicInputBlocks_length]

/-- The exact symbolic private-variable footprint of this leaf. -/
def recipeCount {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) : Nat :=
  Formal.recipeCount (actions interface offset)

/-- The fixed profile compiles the digest-only statement prefix to exactly
160,432 private recipe variables. -/
theorem recipeCount_eq {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    recipeCount interface offset = 160432 := by
  exact publicInputActions_recipeCount interface offset

@[simp] theorem program_recipes_length {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (program interface offset).recipes.length = 160432 := by
  change (Formal.compile offset Hash.zeroE
    (actions interface offset)).recipes.length = 160432
  rw [Formal.compile_recipes_length]
  exact recipeCount_eq interface offset

/-- Layout may allocate exactly this private interval and no boundary copy. -/
theorem localLength_eq {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    localLength (Circuit.ops (circuit interface).main offset) = 160432 := by
  rw [circuit_ops, opsAt_localLength]
  exact program_recipes_length interface offset

/-- The owned statement leaf emits one witness operation and no copy
assertion. -/
theorem operations_length {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (Circuit.ops (circuit interface).main offset).length = 1 := by
  rfl

/-- One row per causal recipe; the compiled final state is reused directly. -/
theorem flatConstraints_length {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (flatConstraints (Circuit.ops (circuit interface).main offset)).length =
      160432 := by
  rw [circuit_ops, flatConstraints_opsAt, recipeConstraints_length]
  exact program_recipes_length interface offset

/-- The verifier claim block is visibly `Eval_K ++ Eval_A`; no matrix
coordinate can inhabit the Pad prefix. -/
theorem verifierClaimWords_eq_eval_K_append_eval_A
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    verifierClaimWords interface offset =
      ((canonicalPadCoordinates productionShape).flatMap fun coordinate =>
        serializeKExpr
          (((interface.running offset).evaluation coordinate.running).eval_K
            coordinate.coefficient)) ++
      (canonicalMatrixCoordinates productionShape).flatMap fun coordinate =>
        serializeKExpr
          (((interface.running offset).evaluation coordinate.running).eval_A
            coordinate.matrix coordinate.coefficient) := by
  rfl

def evalState (env : Env) (state : Layer.EState) :
    NightstreamFPrime.Lifecycle.Transcript.State :=
  List.ofFn (Layer.evalState env state)

private theorem eval_zeroE_eq_initialState (env : Env) :
    evalState env Hash.zeroE =
      NightstreamFPrime.Lifecycle.Transcript.initialState := by
  unfold evalState NightstreamFPrime.Lifecycle.Transcript.initialState
    NightstreamFPrime.Spec.Poseidon2.zeroState
  change List.ofFn (fun _ : Fin 8 => (0 : F)) = List.replicate 8 0
  norm_num [List.ofFn_succ]
  rfl

/-- Exact parent coverage for the statement transcript boundary. The
symbolic public values are evaluated into the production `Running` and
`Fresh` types; no caller supplies the resulting state equality. -/
theorem spec_implies_keyInitialState
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (specification : SpecHolds interface offset env) :
    evalState env (finalState interface offset) =
      let running := evalRunning (interface.running offset) env
      let fresh := evalFresh (interface.fresh offset) env
      let key := ProductionKey.key relation ajtai
      key.oracle.transcript.initialState
        ({ priorState := key.publicInputState running fresh
           input := (key.statement running fresh).verifierInput key.lift } :
          NightstreamFPrime.Spec.Folding.PiCCS.TranscriptReplay.Statement K
            NightstreamFPrime.Lifecycle.Transcript.State productionShape) := by
  let running := evalRunning (interface.running offset) env
  let fresh := evalFresh (interface.fresh offset) env
  let key := ProductionKey.key relation ajtai
  let input := (key.statement running fresh).verifierInput key.lift
  let context : NightstreamFPrime.Spec.Folding.PiCCS.TranscriptReplay.Statement K
      NightstreamFPrime.Lifecycle.Transcript.State productionShape :=
    { priorState := key.publicInputState running fresh, input := input }
  have trace := specification
  change Formal.TraceHolds
      (evalState env Hash.zeroE)
      ((actions interface offset).map (Formal.Action.eval env))
      (evalState env (finalState interface offset)) at trace
  rw [actions_eq, List.map_append] at trace
  change Formal.TraceHolds
      (Absorb.reference (evalState env Hash.zeroE)
        (Hash.evalList env
          (constantWords
            NightstreamFPrime.Lifecycle.Transcript.piCcsDigestDomainTag)))
      (((absorbedBlocks interface offset).map absorbBlock).map
        (Formal.Action.eval env))
      (evalState env (finalState interface offset)) at trace
  rw [constantWords_eval, eval_zeroE_eq_initialState,
    reference_eq_absorb] at trace
  rw [absorbBlockActions_eval] at trace
  have actionListEq :
      (absorbedBlocks interface offset).map (fun words =>
        Formal.ValueAction.absorb
          (NightstreamFPrime.Lifecycle.block (Hash.evalList env words))) =
      ((absorbedBlocks interface offset).map (Hash.evalList env)).map
        (fun words => Formal.ValueAction.absorb
          (NightstreamFPrime.Lifecycle.block words)) := by
    rw [List.map_map]
    rfl
  rw [actionListEq] at trace
  have folded := (traceHolds_absorbBlocks_iff
    (NightstreamFPrime.Lifecycle.Transcript.absorb
      NightstreamFPrime.Lifecycle.Transcript.initialState
      NightstreamFPrime.Lifecycle.Transcript.piCcsDigestDomainTag)
    (evalState env (finalState interface offset))
    ((absorbedBlocks interface offset).map (Hash.evalList env))).mp trace
  have blocksEq := absorbedBlocks_eval interface offset env
  rw [blocksEq] at folded
  have publicStateEq := ProductionKey.key_publicInputState_eq relation ajtai
    running fresh
  have oracleStateEq := ProductionKey.key_oracle_initialState_eq relation ajtai
    context
  calc
    evalState env (finalState interface offset) =
        NightstreamFPrime.Lifecycle.Transcript.absorbBlocks
          (NightstreamFPrime.Lifecycle.Transcript.absorb
            NightstreamFPrime.Lifecycle.Transcript.initialState
            NightstreamFPrime.Lifecycle.Transcript.piCcsDigestDomainTag)
          (ProductionKey.publicInputBlocks running fresh) :=
      folded.symm
    _ = key.publicInputState running fresh := publicStateEq.symm
    _ = key.oracle.transcript.initialState context := oracleStateEq.symm
    _ = _ := by rfl

end NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption

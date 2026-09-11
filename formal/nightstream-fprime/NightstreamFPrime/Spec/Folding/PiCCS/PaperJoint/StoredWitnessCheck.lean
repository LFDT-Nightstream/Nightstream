import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CheckedWitnessExtraction

/-!
SuperNeo v1.1, Definition 20 and Appendix B.2, extractor step 3.
Checks the public fixed-width response and the complete ambient relation on
the same stored arrays. Commitment, public prefix, strict ambient norm, Pad,
and every matrix coefficient are checked against verifier-owned inputs.

Finite folds read the arrays without building a column-index list. The MLE
recursion evaluates rows without building a Boolean table. These are value
refinements only: key and matrix access, commitment computation, public
response evaluation, and their execution clocks remain separate obligations.
No opening-validity or cryptographic premise is used by this checker.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StoredWitnessCheck

open NightstreamFPrime.Spec
open StrongReduction ConcreteCarrier FullOutputCoordinates PaperLinearAlgebra
open StoredWitnessProjection
open CheckedWitnessExtraction (openingMaps)

/-- Check each finite coordinate without constructing an index list. -/
def allFin {count : Nat} (test : Fin count → Bool) : Bool :=
  Fin.foldl count (fun accepted index => accepted && test index) true

private theorem fold_all_eq_true : ∀ {count : Nat}
    (test : Fin count → Bool) (initial : Bool),
    Fin.foldl count (fun accepted index => accepted && test index) initial = true ↔
      initial = true ∧ ∀ index, test index = true
  | 0, _, _ => by simp
  | _ + 1, test, initial => by
      rw [Fin.foldl_succ, fold_all_eq_true]
      simp only [Bool.and_eq_true, Fin.forall_fin_succ, and_assoc]

/-- The finite fold accepts exactly when every coordinate accepts. -/
theorem allFin_eq_true {count : Nat} (test : Fin count → Bool) :
    allFin test = true ↔ ∀ index, test index = true := by
  simp only [allFin, fold_all_eq_true, true_and]

/-- The same ordered dot product as `matrixVectorAt`, with no index list. -/
def matrixRow {arity columns : Nat}
    (matrix : BooleanMatrix F arity columns) (assignment : Vector F columns)
    (vertex : BooleanVertex arity) : F :=
  Fin.foldl columns (fun accumulated column =>
    baseOps.add accumulated (baseOps.mul (matrix vertex column) (assignment.get column)))
    baseOps.zero

/-- The streaming dot product equals the existing mathematical row value. -/
theorem matrixRow_eq {arity columns : Nat}
    (matrix : BooleanMatrix F arity columns) (assignment : Vector F columns)
    (vertex : BooleanVertex arity) :
    matrixRow matrix assignment vertex =
      matrixVectorAt baseOps matrix assignment.get vertex := by
  rw [matrixRow, Fin.foldl_eq_finRange_foldl]
  rfl

/-- Fuse tabulation and interpolation; only the current low/high values live. -/
def evaluateRows : {arity : Nat} →
    (BooleanVertex arity → K) → List K → K
  | 0, values, [] => values .nil
  | _ + 1, values, coordinate :: coordinates =>
      let low := evaluateRows (fun tail => values (.cons false tail)) coordinates
      let high := evaluateRows (fun tail => values (.cons true tail)) coordinates
      extensionOps.add low (extensionOps.mul coordinate (extensionOps.sub high low))
  | _, _, _ => extensionOps.zero

/-- Streaming interpolation equals the complete table's evaluation. -/
theorem evaluateRows_eq {arity : Nat}
    (values : BooleanVertex arity → K) (coordinates : List K) :
    evaluateRows values coordinates =
      (BooleanTable.tabulate values).evaluateCoordinates extensionOps coordinates := by
  induction arity generalizing coordinates with
  | zero => cases coordinates <;> rfl
  | succ arity inductionHypothesis =>
      cases coordinates <;>
        simp only [evaluateRows, BooleanTable.tabulate,
          BooleanTable.evaluateCoordinates, inductionHypothesis]

variable {Commitment : Type*} {shape : Shape} {carrier : Phi81Relation.Shape}
  {blockCount : Nat}

/-- Compute all separate Pad and matrix evaluations from the stored rows. -/
def evaluations
    (statement : Statement K Commitment (Phi81Relation.PublicInput carrier)
      shape carrier.carrierWidth blockCount baseOps)
    (probe : Probe K shape) (stored : StoredWitness shape carrier) : FullOutput K shape where
  padCoordinate := fun source coefficient =>
    evaluateRows (fun vertex => K.embed (matrixRow
      (statement.matrixSource.coefficientMatrixOf baseOps
        (fun row column => statement.cubeLayout.paddedIdentityEntry
          baseOps.zero baseOps.one row column) coefficient)
      (stored.get source) vertex)) probe.coins.roundPoint.coordinates
  matrixCoordinate := fun source matrix coefficient =>
    evaluateRows (fun vertex => K.embed (matrixRow
      (statement.matrixSource.coefficientMatrix baseOps matrix coefficient)
      (stored.get source) vertex)) probe.coins.roundPoint.coordinates

private theorem fullOutput_ext {left right : FullOutput K shape}
    (pad : left.padCoordinate = right.padCoordinate)
    (matrix : left.matrixCoordinate = right.matrixCoordinate) : left = right := by
  cases left
  cases right
  cases pad
  cases matrix
  rfl

/-- Streaming row evaluation is exactly the existing complete paper output. -/
theorem evaluations_eq_honestAt
    (statement : Statement K Commitment (Phi81Relation.PublicInput carrier)
      shape carrier.carrierWidth blockCount baseOps)
    (probe : Probe K shape) (stored : StoredWitness shape carrier) :
    evaluations statement probe stored =
      FullOutput.honestAt baseOps extensionOps K.embed
        (statement.sourceConnectedInputs (view stored)) probe.coins.roundPoint := by
  apply fullOutput_ext
  · funext source coefficient
    simp only [evaluations, evaluateRows_eq, matrixRow_eq]
    rfl
  · funext source matrix coefficient
    simp only [evaluations, evaluateRows_eq, matrixRow_eq]
    rfl

/-- Opening checks read the exact array used by the evaluation checks. -/
def openingCheck [DecidableEq Commitment]
    (commit : Phi81Relation.Assignment carrier → Commitment) (params : GlobalParams)
    (statement : Statement K Commitment (Phi81Relation.PublicInput carrier)
      shape carrier.carrierWidth blockCount baseOps)
    (stored : StoredWitness shape carrier) (source : Fin shape.sourceCount) : Bool :=
  decide (commit (stored.get source).get = statement.commitments source) &&
    (allFin (fun column : Fin carrier.publicWidth =>
      decide ((stored.get source).get (carrier.publicColumn column) =
        statement.publicInputs source column)) &&
      allFin (fun column : Fin carrier.carrierWidth =>
        decide (centeredMagnitude ((stored.get source).get column) < params.ambientBound)))

private theorem openingCheck_eq_true_iff [DecidableEq Commitment]
    (commit : Phi81Relation.Assignment carrier → Commitment) (params : GlobalParams)
    (statement : Statement K Commitment (Phi81Relation.PublicInput carrier)
      shape carrier.carrierWidth blockCount baseOps)
    (probe : Probe K shape) (stored : StoredWitness shape carrier) (source : Fin shape.sourceCount) :
    openingCheck commit params statement stored source = true ↔
      Opening.Holds (paperRelationSemantics (shape := shape) (blockCount := blockCount)
        baseOps extensionOps K.embed (openingMaps commit))
        (Folding.PiRLC.PaperCorrections.correctedAmbientBoundFor params)
        (statement.publicOutput probe source).commitment
        (statement.publicOutput probe source).publicInput ((view stored).assignments source) := by
  simp only [openingCheck, Bool.and_eq_true, allFin_eq_true, decide_eq_true_eq,
    Opening.Holds, paperRelationSemantics, openingMaps, Statement.publicOutput, view,
    Folding.PiRLC.PaperCorrections.correctedAmbientBoundFor]
  exact and_congr Iff.rfl (and_congr ⟨funext, fun equal => congrFun equal⟩ Iff.rfl)

/-- Check every coefficient of both evaluation families for one source. -/
def evaluationCheck (expected received : FullOutput K shape)
    (source : Fin shape.sourceCount) : Bool :=
  allFin (fun coefficient : Fin shape.coefficientCount =>
    decide (expected.padCoordinate source coefficient = received.padCoordinate source coefficient)) &&
    allFin (fun matrix : Fin shape.matrixCount =>
      allFin (fun coefficient : Fin shape.coefficientCount =>
        decide (expected.matrixCoordinate source matrix coefficient =
          received.matrixCoordinate source matrix coefficient)))

private theorem family_ext {left right : EvaluationFamily K shape}
    (pad : left.pad = right.pad) (matrix : left.matrix = right.matrix) : left = right := by
  cases left
  cases right
  cases pad
  cases matrix
  rfl

private theorem evaluationCheck_eq_true_iff (expected received : FullOutput K shape)
    (source : Fin shape.sourceCount) :
    evaluationCheck expected received source = true ↔
      #[({ pad := expected.padCoordinate source,
           matrix := expected.matrixCoordinate source } : EvaluationFamily K shape)] =
      #[({ pad := received.padCoordinate source,
           matrix := received.matrixCoordinate source } : EvaluationFamily K shape)] := by
  simp only [evaluationCheck, Bool.and_eq_true, allFin_eq_true, decide_eq_true_eq]
  constructor
  · rintro ⟨pad, matrix⟩
    exact congrArg (fun family : EvaluationFamily K shape => #[family])
      (family_ext (funext pad) (funext fun index => funext (matrix index)))
  · intro equal
    have families :
        ({ pad := expected.padCoordinate source,
           matrix := expected.matrixCoordinate source } : EvaluationFamily K shape) =
        ({ pad := received.padCoordinate source,
           matrix := received.matrixCoordinate source } : EvaluationFamily K shape) := by
      have lists := congrArg Array.toList equal
      simpa using lists
    exact ⟨fun coefficient => congrFun (congrArg EvaluationFamily.pad families) coefficient,
      fun matrix coefficient =>
        congrFun (congrFun (congrArg EvaluationFamily.matrix families) matrix) coefficient⟩

/-- Exact corrected-ambient check, including all openings and evaluations. -/
def ambientCheck [DecidableEq Commitment]
    (commit : Phi81Relation.Assignment carrier → Commitment) (params : GlobalParams)
    (statement : Statement K Commitment (Phi81Relation.PublicInput carrier)
      shape carrier.carrierWidth blockCount baseOps)
    (probe : Probe K shape) (stored : StoredWitness shape carrier) : Bool :=
  let expected := evaluations statement probe stored
  allFin fun source => openingCheck commit params statement stored source &&
    evaluationCheck expected probe.response.fullOutput source

/-- No assumed opening truth: acceptance is equivalent to the full ambient relation. -/
theorem ambientCheck_eq_true_iff [DecidableEq Commitment]
    (commit : Phi81Relation.Assignment carrier → Commitment) (params : GlobalParams)
    (statement : Statement K Commitment (Phi81Relation.PublicInput carrier)
      shape carrier.carrierWidth blockCount baseOps)
    (probe : Probe K shape) (stored : StoredWitness shape carrier) :
    ambientCheck commit params statement probe stored = true ↔
      AmbientOutputHolds extensionOps K.embed (openingMaps commit) params statement probe (view stored) := by
  rw [ambientCheck, allFin_eq_true]
  unfold AmbientOutputHolds
  apply forall_congr'
  intro source
  rw [Bool.and_eq_true, openingCheck_eq_true_iff commit params statement probe,
    evaluationCheck_eq_true_iff, evaluations_eq_honestAt]
  change (_ ∧ _) ↔ (_ ∧ True ∧ _)
  rw [true_and]
  rfl

/-- Execute the existing public verifier before checking the stored witness. -/
def check [DecidableEq Commitment]
    (commit : Phi81Relation.Assignment carrier → Commitment) (params : GlobalParams)
    (statement : Statement K Commitment (Phi81Relation.PublicInput carrier)
      shape carrier.carrierWidth blockCount baseOps)
    (width : Nat) (candidate : Probe K shape × StoredWitness shape carrier) : Bool :=
  ProtocolPolynomial.FixedWidth.check extensionOps width (statement.verifierInput K.embed)
    candidate.1.coins.alpha candidate.1.coins.gamma candidate.1.coins.roundPoint
    (statement.projectOutput candidate.1.response.fullOutput) candidate.1.response.rounds &&
      ambientCheck commit params statement candidate.1 candidate.2

/-- The Boolean part of `finishStored_source_iff` and `run_source_iff`'s
`checkCorrect` premise. A costed implementation must still refine this call. -/
theorem check_eq_true_iff [DecidableEq Commitment]
    (commit : Phi81Relation.Assignment carrier → Commitment) (params : GlobalParams)
    (statement : Statement K Commitment (Phi81Relation.PublicInput carrier)
      shape carrier.carrierWidth blockCount baseOps)
    (width : Nat) (probe : Probe K shape) (stored : StoredWitness shape carrier) :
    check commit params statement width (probe, stored) = true ↔
      probe.FixedWidthAccepted extensionOps K.embed statement width ∧
        AmbientOutputHolds extensionOps K.embed (openingMaps commit) params statement probe (view stored) := by
  simp only [check, Bool.and_eq_true, ambientCheck_eq_true_iff, Probe.FixedWidthAccepted]

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StoredWitnessCheck

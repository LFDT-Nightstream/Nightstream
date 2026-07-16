import Nightstream.SuperNeo.Relations
import Nightstream.SuperNeo.Concrete.Algebra

/-!
Concrete, executable relation model using the SuperNeo algebraic vocabulary.

The CCS predicate checks a well-formed collection of Goldilocks matrices and a
sparse polynomial row by row. The current CE model first computes the scalar
row vector `M * z`, then packs consecutive rows into coefficient lanes before
multilinear evaluation. This is not the paper Phi81 coefficient-matrix
construction; `Concrete.Necessity.Phi81OutputMismatch` gives an exact
counterexample to identifying the two interpretations.
Commitment and model-level public-input authority come from the concrete Ajtai
action and caller-parameterized prefix projection in `Algebra.lean`. This file
does not prove that an arbitrary prefix width satisfies the paper precondition
`n_F,in = d * n_R,in`, nor that `take publicWidth` refines Definition 13's
ring-module `L_in`. Those are separate production-refinement obligations.

Non-goals: optimized Rust arithmetic refinement, sparse-matrix serialization,
cryptographic Ajtai binding, and sumcheck soundness.
-/

namespace Nightstream.SuperNeo.Concrete

abbrev Assignment := List F
abbrev PublicInput := List F
abbrev Point := List K
abbrev Evaluation := RingK
abbrev Matrix := List (List F)

def dotF : List F → List F → F
  | a :: as, b :: bs => a * b + dotF as bs
  | _, _ => 0

def matrixVector (matrix : Matrix) (z : Assignment) : List F :=
  matrix.map (fun row => dotF row z)

structure Monomial where
  coefficient : F
  exponents : List Nat
deriving DecidableEq, Repr

abbrev Polynomial := List Monomial

def evalPowers : List F → List Nat → F
  | x :: xs, exponent :: exponents => x ^ exponent * evalPowers xs exponents
  | [], [] => 1
  | _, _ => 0

def evalMonomial (point : List F) (term : Monomial) : F :=
  term.coefficient * evalPowers point term.exponents

def evalPolynomial (polynomial : Polynomial) (point : List F) : F :=
  polynomial.foldl (fun acc term => acc + evalMonomial point term) 0

structure Structure where
  matrices : List Matrix
  polynomial : Polynomial
  rows : Nat
  columns : Nat
  /-- `log₂` of the zero-padded ring-column domain used by CE. -/
  pointDimension : Nat
deriving DecidableEq, Repr

def MatrixWellFormed (rows columns : Nat) (matrix : Matrix) : Prop :=
  matrix.length = rows ∧ ∀ row ∈ matrix, row.length = columns

def Structure.WellFormed (s : Structure) : Prop :=
  s.matrices ≠ [] ∧
  (∀ matrix ∈ s.matrices, MatrixWellFormed s.rows s.columns matrix) ∧
  (∀ term ∈ s.polynomial, term.exponents.length = s.matrices.length) ∧
  s.rows ≤ ringDegree * 2 ^ s.pointDimension

def rowPoint (s : Structure) (z : Assignment) (row : Nat) : List F :=
  s.matrices.map (fun matrix => (matrixVector matrix z).getD row 0)

/-- Rust's row-wise CCS zero check, with all shape obligations explicit. -/
def ccsSatisfied (s : Structure) (z : Assignment) : Prop :=
  s.WellFormed ∧ z.length = s.columns ∧
  ∀ row, row < s.rows → evalPolynomial s.polynomial (rowPoint s z row) = 0

def mleRound (challenge : K) : List K → List K
  | [] => []
  | [a] => [K.mul a (K.sub K.one challenge)]
  | a :: b :: rest =>
      K.add (K.mul a (K.sub K.one challenge)) (K.mul b challenge) ::
        mleRound challenge rest

def padK (target : Nat) (values : List K) : List K :=
  (values ++ List.replicate (target - values.length) K.zero).take target

/-- Multilinear extension of a zero-padded vector at `point`. -/
def mleEval (values : List F) (point : Point) : K :=
  let initial := padK (2 ^ point.length) (values.map K.embed)
  (point.foldl (fun layer challenge => mleRound challenge layer) initial).headD K.zero

def coefficientLane (values : List F) (point : Point)
    (rho : Fin ringDegree) : List F :=
  (List.range (2 ^ point.length)).map
    (fun block => values.getD (block * ringDegree + rho.val) 0)

/-- Current ring-valued CE opening: compute scalar matrix rows, split those
rows by residue modulo `ringDegree`, then evaluate one MLE per residue. This is
deliberately not documented as the paper Phi81 CE construction. -/
def ringMle (values : List F) (point : Point) : RingK :=
  fun rho => mleEval (coefficientLane values point rho) point

def evaluationPointValid (s : Structure) (point : Point) : Prop :=
  s.WellFormed ∧ point.length = s.pointDimension

def matrixEvaluations (s : Structure) (z : Assignment) (point : Point) :
    Array Evaluation :=
  (s.matrices.map (fun matrix => ringMle (matrixVector matrix z) point)).toArray

structure Context where
  publicWidth : Nat
  ajtaiKey : AjtaiKey

/-- Executable operations plugged into the generic relation interface. A
production SuperNeo instantiation additionally needs the aligned `L_in`
refinement described in the module contract. -/
def relationSemantics (context : Context) :
    RelationSemantics Structure Assignment PublicInput Point Evaluation Commitment where
  commit := ajtaiCommit context.ajtaiKey
  projectPublicInput := projectPublicInput context.publicWidth
  normBounded := normBounded
  ccsSatisfied := ccsSatisfied
  evaluationPointValid := evaluationPointValid
  evaluations := matrixEvaluations

abbrev CCSStatement := CCS.Instance Structure PublicInput Commitment
abbrev CEStatement := CE.Instance Structure PublicInput Point Evaluation Commitment

/-- Exact expansion of concrete CCS membership. -/
theorem ccsMembership_iff (context : Context) (params : GlobalParams)
    (statement : CCSStatement) (z : Assignment) :
    CCS.Holds (relationSemantics context) params statement z ↔
      ajtaiCommit context.ajtaiKey z = statement.commitment ∧
      projectPublicInput context.publicWidth z = statement.publicInput ∧
      normBounded (statement.stage.bound params) z ∧
      ccsSatisfied statement.constraintSystem z := by
  simp [CCS.Holds, Opening.Holds, relationSemantics, and_assoc]

/-- Exact expansion of concrete CE membership, including point-domain shape. -/
theorem ceMembership_iff (context : Context) (params : GlobalParams)
    (statement : CEStatement) (z : Assignment) :
    CE.Holds (relationSemantics context) params statement z ↔
      ajtaiCommit context.ajtaiKey z = statement.commitment ∧
      projectPublicInput context.publicWidth z = statement.publicInput ∧
      normBounded (statement.stage.bound params) z ∧
      evaluationPointValid statement.constraintSystem statement.point ∧
      matrixEvaluations statement.constraintSystem z statement.point =
        statement.evaluations := by
  simp [CE.Holds, Opening.Holds, relationSemantics, and_assoc]

def canonicalCCSStatement (context : Context) (system : Structure)
    (stage : NormStage) (z : Assignment) : CCSStatement where
  constraintSystem := system
  commitment := ajtaiCommit context.ajtaiKey z
  publicInput := projectPublicInput context.publicWidth z
  stage := stage

def canonicalCEStatement (context : Context) (system : Structure)
    (stage : NormStage) (point : Point) (z : Assignment) : CEStatement where
  constraintSystem := system
  commitment := ajtaiCommit context.ajtaiKey z
  publicInput := projectPublicInput context.publicWidth z
  point := point
  evaluations := matrixEvaluations system z point
  stage := stage

theorem canonicalCCS_holds (context : Context) (params : GlobalParams)
    (system : Structure) (stage : NormStage) (z : Assignment)
    (hnorm : normBounded (stage.bound params) z)
    (hsatisfied : ccsSatisfied system z) :
    CCS.Holds (relationSemantics context) params
      (canonicalCCSStatement context system stage z) z := by
  exact ⟨⟨rfl, rfl, hnorm⟩, hsatisfied⟩

theorem canonicalCE_holds (context : Context) (params : GlobalParams)
    (system : Structure) (stage : NormStage) (point : Point) (z : Assignment)
    (hnorm : normBounded (stage.bound params) z)
    (hpoint : evaluationPointValid system point) :
    CE.Holds (relationSemantics context) params
      (canonicalCEStatement context system stage point z) z := by
  exact ⟨⟨rfl, rfl, hnorm⟩, hpoint, rfl⟩

theorem ccs_rejects_wrong_commitment
    (context : Context) (params : GlobalParams) (statement : CCSStatement)
    (z : Assignment)
    (hwrong : statement.commitment ≠ ajtaiCommit context.ajtaiKey z) :
    ¬ CCS.Holds (relationSemantics context) params statement z := by
  intro h
  exact hwrong h.1.1.symm

theorem ccs_rejects_wrong_public_input
    (context : Context) (params : GlobalParams) (statement : CCSStatement)
    (z : Assignment)
    (hwrong : statement.publicInput ≠ projectPublicInput context.publicWidth z) :
    ¬ CCS.Holds (relationSemantics context) params statement z := by
  intro h
  exact hwrong h.1.2.1.symm

theorem ce_rejects_invalid_point
    (context : Context) (params : GlobalParams) (statement : CEStatement)
    (z : Assignment)
    (hinvalid : ¬ evaluationPointValid statement.constraintSystem statement.point) :
    ¬ CE.Holds (relationSemantics context) params statement z := by
  intro h
  exact hinvalid h.2.1

theorem ce_rejects_wrong_evaluations
    (context : Context) (params : GlobalParams) (statement : CEStatement)
    (z : Assignment)
    (hwrong : statement.evaluations ≠
      matrixEvaluations statement.constraintSystem z statement.point) :
    ¬ CE.Holds (relationSemantics context) params statement z := by
  intro h
  exact hwrong h.2.2.symm

end Nightstream.SuperNeo.Concrete

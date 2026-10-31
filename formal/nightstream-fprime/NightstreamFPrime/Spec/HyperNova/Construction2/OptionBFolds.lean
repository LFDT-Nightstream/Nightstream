import NightstreamFPrime.Spec.Folding.Light.Verifier
import NightstreamFPrime.Spec.Folding.Refresh.Verifier
import NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepth

/-! Option B's two fold verifiers as one `BoundedDepth.Folds`. A slot holds one accumulator of
the refresh shape. The light shape is the refresh shape with one running claim, so the two folds
have one CCS shape (R1a): the same `cubeVariables`, `matrixCount` and `coefficientCount`. A claim
moves between the shapes by rebuilding its structure, and the light fold reads the slot's
accumulator as its single running claim. A fold proof names its fold; the other fold rejects
it. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.HyperNova.Construction2.OptionBFolds

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.MatrixCoefficientSource
open NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive (Fresh Running)

universe uCommitment uPublicInput uScalar uState

/-- The light fold's shape: the refresh shape with one running claim. -/
def lightShape (shape : Shape) : Shape := { shape with runningCount := 1 }

variable {Commitment : Type uCommitment} {PublicInput : Type uPublicInput}
  {Scalar : Type uScalar} {State : Type uState} {shape : Shape}
  {columns blockCount width degreeBound : Nat}

def evaluationsToLight (family : EvaluationFamily K shape) :
    EvaluationFamily K (lightShape shape) :=
  ⟨family.pad, family.matrix⟩

def evaluationsFromLight (family : EvaluationFamily K (lightShape shape)) :
    EvaluationFamily K shape :=
  ⟨family.pad, family.matrix⟩

def sourceFromLight (source : RelationSource (lightShape shape) columns blockCount) :
    RelationSource shape columns blockCount :=
  ⟨source.cubeLayout, ⟨source.matrixSource.columnLayout, source.matrixSource.matrices,
    source.matrixSource.constraintPolynomial, source.matrixSource.kernel⟩⟩

def freshToLight (fresh : Fresh Commitment PublicInput shape) :
    Fresh Commitment PublicInput (lightShape shape) :=
  ⟨fresh.commitments, fresh.publicInputs⟩

/-- A slot's running claim: one accumulator of the refresh shape. -/
abbrev Claim (Commitment : Type uCommitment) (PublicInput : Type uPublicInput) (shape : Shape)
    (columns blockCount : Nat) :=
  Refresh.Accumulator Commitment PublicInput shape columns blockCount

/-- The slot's accumulator as the light fold's running claim, when it carries one evaluation
family. -/
def toLightRunning (claim : Claim Commitment PublicInput shape columns blockCount) :
    Option (Running K Commitment PublicInput (lightShape shape)) :=
  if size : claim.evaluations.size = 1 then
    some { point := claim.point
           commitments := fun _ => claim.commitment
           publicInputs := fun _ => claim.publicInput
           evaluations := fun _ => evaluationsToLight (claim.evaluations[0]'(by omega)) }
  else none

/-- The light fold's output as a slot claim of the refresh shape. -/
def fromLight (output : CE.Instance (RelationSource (lightShape shape) columns blockCount)
      PublicInput (CubePoint K (lightShape shape).cubeVariables)
      (EvaluationFamily K (lightShape shape)) Commitment) :
    Claim Commitment PublicInput shape columns blockCount where
  constraintSystem := sourceFromLight output.constraintSystem
  commitment := output.commitment
  publicInput := output.publicInput
  point := output.point
  evaluations := output.evaluations.map evaluationsFromLight
  stage := output.stage

/-- The verifier keys of one slot: its light fold and its refresh fold. -/
structure Keys (Commitment : Type uCommitment) (PublicInput : Type uPublicInput)
    (Scalar : Type uScalar) (State : Type uState) (shape : Shape)
    (columns blockCount width degreeBound : Nat) where
  light : Light.Key Commitment PublicInput Scalar State (lightShape shape) columns blockCount width
  refresh : Nifs.PaperNonInteractive.Key K Commitment PublicInput Scalar State shape columns
    blockCount degreeBound

/-- A fold proof names the fold it is for. -/
inductive FoldProof (Commitment : Type uCommitment) (shape : Shape) (width degreeBound : Nat) where
  | light (proof : Light.Proof (lightShape shape) width)
  | refresh (proof : Refresh.Proof Commitment shape degreeBound)

/-- Option B's fold verifiers: `Light.verify` on the slot's accumulator read as one running
claim, and `Refresh.verify` on the accumulator itself. -/
def folds :
    BoundedDepth.Folds (Keys Commitment PublicInput Scalar State shape columns blockCount width
        degreeBound)
      (Claim Commitment PublicInput shape columns blockCount) (Fresh Commitment PublicInput shape)
      (FoldProof Commitment shape width degreeBound)
  | .light, keys, claim, fresh, .light proof =>
      (toLightRunning claim).bind fun running =>
        (Light.verify keys.light running (freshToLight fresh) proof).map fromLight
  | .refresh, keys, claim, fresh, .refresh proof => Refresh.verify keys.refresh claim fresh proof
  | _, _, _, _, _ => none

theorem light_eq_some_iff
    (keys : Keys Commitment PublicInput Scalar State shape columns blockCount width degreeBound)
    (claim : Claim Commitment PublicInput shape columns blockCount)
    (fresh : Fresh Commitment PublicInput shape) (proof : Light.Proof (lightShape shape) width)
    (next : Claim Commitment PublicInput shape columns blockCount) :
    folds .light keys claim fresh (.light proof) = some next ↔
      ∃ running output, toLightRunning claim = some running ∧
        Light.verify keys.light running (freshToLight fresh) proof = some output ∧
        next = fromLight output := by
  simp only [folds, Option.bind_eq_some_iff, Option.map_eq_some_iff]
  constructor
  · rintro ⟨running, readEq, output, verified, rfl⟩
    exact ⟨running, output, readEq, verified, rfl⟩
  · rintro ⟨running, output, readEq, verified, rfl⟩
    exact ⟨running, readEq, output, verified, rfl⟩

theorem refresh_eq
    (keys : Keys Commitment PublicInput Scalar State shape columns blockCount width degreeBound)
    (claim : Claim Commitment PublicInput shape columns blockCount)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Refresh.Proof Commitment shape degreeBound) :
    folds .refresh keys claim fresh (.refresh proof) =
      Refresh.verify keys.refresh claim fresh proof :=
  rfl

theorem light_rejects_refresh_proof
    (keys : Keys Commitment PublicInput Scalar State shape columns blockCount width degreeBound)
    (claim : Claim Commitment PublicInput shape columns blockCount)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Refresh.Proof Commitment shape degreeBound) :
    folds .light keys claim fresh (.refresh proof) = none :=
  rfl

theorem refresh_rejects_light_proof
    (keys : Keys Commitment PublicInput Scalar State shape columns blockCount width degreeBound)
    (claim : Claim Commitment PublicInput shape columns blockCount)
    (fresh : Fresh Commitment PublicInput shape) (proof : Light.Proof (lightShape shape) width) :
    folds .refresh keys claim fresh (.light proof) = none :=
  rfl

/-- A light output with one evaluation family is read back as the light running claim it
describes, so light folds chain without loss. -/
theorem toLightRunning_fromLight
    (output : CE.Instance (RelationSource (lightShape shape) columns blockCount) PublicInput
      (CubePoint K (lightShape shape).cubeVariables) (EvaluationFamily K (lightShape shape))
      Commitment)
    (size : output.evaluations.size = 1) :
    toLightRunning (fromLight output) =
      some { point := output.point
             commitments := fun _ => output.commitment
             publicInputs := fun _ => output.publicInput
             evaluations := fun _ => output.evaluations[0]'(by omega) } := by
  have mapped : (fromLight output).evaluations.size = 1 := by
    simp [fromLight, size]
  simp only [toLightRunning, mapped, ↓reduceDIte, Option.some.injEq]
  simp [fromLight, evaluationsToLight, evaluationsFromLight]

/-- A slot claim that holds under the refresh key's relation has one evaluation family
(`piRlcEvaluationsSize`), so the light fold can always read a valid slot claim. -/
theorem toLightRunning_isSome
    (keys : Keys Commitment PublicInput Scalar State shape columns blockCount width degreeBound)
    (claim : Claim Commitment PublicInput shape columns blockCount)
    (witness : PaperLinearAlgebra.Assignment F columns)
    (holds : CE.Holds keys.refresh.piRlcSemantics keys.refresh.params claim witness) :
    (toLightRunning claim).isSome := by
  have size : claim.evaluations.size = 1 := by
    rw [← holds.2.2]
    exact keys.refresh.piRlcEvaluationsSize _ _ _
  simp [toLightRunning, size]

end NightstreamFPrime.Spec.HyperNova.Construction2.OptionBFolds

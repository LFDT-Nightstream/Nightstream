/-!
Contract: exact verifier-key relation artifact binding.

Owns: the selected profile census and the fail-closed rule that accepts a
carried artifact only when its complete value equals the verifier-owned
artifact. Equality covers source provenance, dimensions, all matrices, the
polynomial, and every key-binding field.

Does not own: JSON decoding, Rust compiler refinement, a deployed verifier
key, Spartan or WHIR soundness, or an on-chain parser.

Assurance tier: model-proved. Rust-origin evidence must separately show that
the implementation uses this exact-validation rule.
-/

set_option autoImplicit false

namespace Nightstream.Assurance.RelationArtifactBinding

universe uMatrix uPolynomial uDigest uSource

/-- Dimensions and padding facts carried by one relation artifact. -/
structure Shape where
  logicalRows : Nat
  assignmentFields : Nat
  paddedRows : Nat
  rowVariables : Nat
  publicStartField : Nat
  publicFields : Nat
  semanticMatrixCount : Nat
  jointMatrixCount : Nat
  polynomialDegree : Nat
deriving Repr, DecidableEq

/-- Complete semantic value of one verifier-key relation artifact. Matrix
entries are data, not digests. The digest fields remain bound metadata. -/
structure Artifact
    (Matrix : Type uMatrix)
    (Polynomial : Type uPolynomial)
    (Digest : Type uDigest)
    (Source : Type uSource) where
  format : String
  schema : Nat
  matrixPayloadEncoding : String
  source : Source
  shape : Shape
  structureDigest : Digest
  matrixDigest : Digest
  ajtaiPublicParametersDigest : Digest
  verifierKeyDigest : Digest
  matrices : List Matrix
  polynomial : Polynomial
deriving Repr, DecidableEq

def artifactFormat : String := "nightstream/verifier-key-relation"

def artifactSchema : Nat := 1

def selectedPaddedRows : Nat := 2 ^ 24

def ringDegree : Nat := 54

/-- Largest complete 54-field ring-column prefix in the selected 24-variable
row domain. This is derived from `2^24`, not selected by convention. -/
def maxAssignmentFields : Nat := (selectedPaddedRows / ringDegree) * ringDegree

def selectedPublicFields : Nat := 270

def selectedSemanticMatrixCount : Nat := 13

def selectedJointMatrixCount : Nat := 14

def selectedPolynomialDegree : Nat := 8

/-- The implicit padded identity has one logical column for each assignment
field. Its width is therefore derived from the authoritative relation. -/
def paddedIdentityWidth (shape : Shape) : Nat := shape.assignmentFields

/-- Stable profile facts. Exact row and assignment counts remain verifier-key
owned values within this selected capacity. -/
def SelectedShape (shape : Shape) : Prop :=
  0 < shape.logicalRows ∧
  0 < shape.assignmentFields ∧
  shape.assignmentFields % ringDegree = 0 ∧
  shape.logicalRows ≤ selectedPaddedRows ∧
  shape.assignmentFields ≤ maxAssignmentFields ∧
  shape.paddedRows = selectedPaddedRows ∧
  shape.rowVariables = 24 ∧
  shape.publicStartField = 0 ∧
  shape.publicFields = selectedPublicFields ∧
  shape.publicFields ≤ shape.assignmentFields ∧
  shape.semanticMatrixCount = selectedSemanticMatrixCount ∧
  shape.jointMatrixCount = selectedJointMatrixCount ∧
  shape.polynomialDegree = selectedPolynomialDegree

/-- Stable format and selected-shape predicate. Source and key identity remain
exact artifact data and are bound by `ExactValidation`. -/
def SelectedProfile
    {Matrix : Type uMatrix}
    {Polynomial : Type uPolynomial}
    {Digest : Type uDigest}
    {Source : Type uSource}
    (artifact : Artifact Matrix Polynomial Digest Source) : Prop :=
  artifact.format = artifactFormat ∧
  artifact.schema = artifactSchema ∧
  artifact.matrices.length = selectedSemanticMatrixCount ∧
  SelectedShape artifact.shape

/-- Fail-closed validation. The carried value has no authority of its own. -/
def ExactValidation
    {Matrix : Type uMatrix}
    {Polynomial : Type uPolynomial}
    {Digest : Type uDigest}
    {Source : Type uSource}
    [DecidableEq Matrix]
    [DecidableEq Polynomial]
    [DecidableEq Digest]
    [DecidableEq Source]
    (authoritative carried : Artifact Matrix Polynomial Digest Source) : Bool :=
  decide (carried = authoritative)

theorem exactValidation_eq_true_iff
    {Matrix : Type uMatrix}
    {Polynomial : Type uPolynomial}
    {Digest : Type uDigest}
    {Source : Type uSource}
    [DecidableEq Matrix]
    [DecidableEq Polynomial]
    [DecidableEq Digest]
    [DecidableEq Source]
    {authoritative carried : Artifact Matrix Polynomial Digest Source} :
    ExactValidation authoritative carried = true ↔ carried = authoritative := by
  simp [ExactValidation]

/-- Acceptance gives equality of the complete artifact, not equality of a
carried digest. -/
theorem accepted_eq_authoritative
    {Matrix : Type uMatrix}
    {Polynomial : Type uPolynomial}
    {Digest : Type uDigest}
    {Source : Type uSource}
    [DecidableEq Matrix]
    [DecidableEq Polynomial]
    [DecidableEq Digest]
    [DecidableEq Source]
    {authoritative carried : Artifact Matrix Polynomial Digest Source}
    (accepted : ExactValidation authoritative carried = true) :
    carried = authoritative :=
  exactValidation_eq_true_iff.mp accepted

/-- A changed complete artifact cannot pass exact validation. -/
theorem changed_rejects
    {Matrix : Type uMatrix}
    {Polynomial : Type uPolynomial}
    {Digest : Type uDigest}
    {Source : Type uSource}
    [DecidableEq Matrix]
    [DecidableEq Polynomial]
    [DecidableEq Digest]
    [DecidableEq Source]
    {authoritative carried : Artifact Matrix Polynomial Digest Source}
    (changed : carried ≠ authoritative) :
    ExactValidation authoritative carried = false := by
  simp [ExactValidation, changed]

/-- Exact acceptance transports the selected profile facts from the verifier
key to the carried artifact. -/
theorem accepted_selectedProfile
    {Matrix : Type uMatrix}
    {Polynomial : Type uPolynomial}
    {Digest : Type uDigest}
    {Source : Type uSource}
    [DecidableEq Matrix]
    [DecidableEq Polynomial]
    [DecidableEq Digest]
    [DecidableEq Source]
    {authoritative carried : Artifact Matrix Polynomial Digest Source}
    (selected : SelectedProfile authoritative)
    (accepted : ExactValidation authoritative carried = true) :
    SelectedProfile carried := by
  rw [accepted_eq_authoritative accepted]
  exact selected

/-- Exact acceptance also fixes the implicit identity width to the
verifier-owned assignment width. -/
theorem accepted_paddedIdentityWidth
    {Matrix : Type uMatrix}
    {Polynomial : Type uPolynomial}
    {Digest : Type uDigest}
    {Source : Type uSource}
    [DecidableEq Matrix]
    [DecidableEq Polynomial]
    [DecidableEq Digest]
    [DecidableEq Source]
    {authoritative carried : Artifact Matrix Polynomial Digest Source}
    (accepted : ExactValidation authoritative carried = true) :
    paddedIdentityWidth carried.shape = authoritative.shape.assignmentFields := by
  rw [accepted_eq_authoritative accepted]
  rfl

end Nightstream.Assurance.RelationArtifactBinding

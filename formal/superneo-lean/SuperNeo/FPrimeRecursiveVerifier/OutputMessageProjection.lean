/-!
Owns: the active `Pi_CCS` output projection, reconstruction predicate, and
injectivity proofs used before `Pi_RLC` challenge derivation.

Does not own: context reconstruction, Poseidon2 hashing, concrete output
serialization, or Rust row refinement.

Emits constraints: no.

Authority boundary: the context and constant-term reconstruction are upstream
authority; the projected `y_ring` and `y_zcol` fields bind the remaining
prover-chosen output.

| Obligation | Lean owner | Guarantee |
|---|---|---|
| Message projection | `piCcsOutputMessage` | Retains active `y_ring` and `y_zcol` |
| Full reconstruction | `reconstructPiCcsOutput` | Rebuilds the claim from fixed context and message |
| Injectivity | `piCcsOutputMessage_injective_on_reconstructed` | Equal messages imply equal reconstructed claims |
-/

namespace SuperNeo.FPrimeRecursiveVerifier

universe u

/-- Fields fixed by the input claims, verifier challenges, header, and schema. -/
structure PiCcsOutputContext
    (Shape Commitment PublicX RowPoint ColumnPoint FoldDigest Sidecars : Type u) where
  shape : Shape
  commitment : Commitment
  publicX : PublicX
  rowPoint : RowPoint
  columnPoint : ColumnPoint
  foldDigest : FoldDigest
  sidecars : Sidecars
deriving DecidableEq

/-- The active, non-canonical-padding output values sent by the prover. -/
structure PiCcsOutputMessage (YRing YZcol : Type u) where
  yRing : YRing
  yZcol : YZcol
deriving DecidableEq

/-- Full verifier-visible output claim before projection. -/
structure PiCcsOutputClaim
    (Shape Commitment PublicX RowPoint ColumnPoint FoldDigest Sidecars
      YRing ConstantTerms YZcol : Type u) where
  context : PiCcsOutputContext
    Shape Commitment PublicX RowPoint ColumnPoint FoldDigest Sidecars
  yRing : YRing
  constantTerms : ConstantTerms
  yZcol : YZcol
deriving DecidableEq

/-- The exact message committed before `Pi_RLC` samples `rho`. -/
def piCcsOutputMessage
    {Shape Commitment PublicX RowPoint ColumnPoint FoldDigest Sidecars
      YRing ConstantTerms YZcol : Type u}
    (claim : PiCcsOutputClaim
      Shape Commitment PublicX RowPoint ColumnPoint FoldDigest Sidecars
      YRing ConstantTerms YZcol) :
    PiCcsOutputMessage YRing YZcol :=
  { yRing := claim.yRing, yZcol := claim.yZcol }

/-- Reconstruct the full claim from fixed context and the committed message. -/
def reconstructPiCcsOutput
    {Shape Commitment PublicX RowPoint ColumnPoint FoldDigest Sidecars
      YRing ConstantTerms YZcol : Type u}
    (constantTerms : YRing → ConstantTerms)
    (context : PiCcsOutputContext
      Shape Commitment PublicX RowPoint ColumnPoint FoldDigest Sidecars)
    (message : PiCcsOutputMessage YRing YZcol) :
    PiCcsOutputClaim
      Shape Commitment PublicX RowPoint ColumnPoint FoldDigest Sidecars
      YRing ConstantTerms YZcol :=
  { context := context
    yRing := message.yRing
    constantTerms := constantTerms message.yRing
    yZcol := message.yZcol }

/-- All fields omitted from the transcript projection obey reconstruction. -/
def PiCcsOutputReconstructed
    {Shape Commitment PublicX RowPoint ColumnPoint FoldDigest Sidecars
      YRing ConstantTerms YZcol : Type u}
    (constantTerms : YRing → ConstantTerms)
    (context : PiCcsOutputContext
      Shape Commitment PublicX RowPoint ColumnPoint FoldDigest Sidecars)
    (claim : PiCcsOutputClaim
      Shape Commitment PublicX RowPoint ColumnPoint FoldDigest Sidecars
      YRing ConstantTerms YZcol) : Prop :=
  claim = reconstructPiCcsOutput constantTerms context (piCcsOutputMessage claim)

@[simp] theorem reconstructPiCcsOutput_message
    {Shape Commitment PublicX RowPoint ColumnPoint FoldDigest Sidecars
      YRing ConstantTerms YZcol : Type u}
    (constantTerms : YRing → ConstantTerms)
    (context : PiCcsOutputContext
      Shape Commitment PublicX RowPoint ColumnPoint FoldDigest Sidecars)
    (message : PiCcsOutputMessage YRing YZcol) :
    piCcsOutputMessage (reconstructPiCcsOutput constantTerms context message) =
      message := by
  rfl

theorem reconstructPiCcsOutput_complete
    {Shape Commitment PublicX RowPoint ColumnPoint FoldDigest Sidecars
      YRing ConstantTerms YZcol : Type u}
    (constantTerms : YRing → ConstantTerms)
    (context : PiCcsOutputContext
      Shape Commitment PublicX RowPoint ColumnPoint FoldDigest Sidecars)
    (message : PiCcsOutputMessage YRing YZcol) :
    PiCcsOutputReconstructed constantTerms context
      (reconstructPiCcsOutput constantTerms context message) := by
  rfl

/-- On reconstructed outputs, equality of the projected message implies equality of the full claim. -/
theorem piCcsOutputMessage_injective_on_reconstructed
    {Shape Commitment PublicX RowPoint ColumnPoint FoldDigest Sidecars
      YRing ConstantTerms YZcol : Type u}
    {constantTerms : YRing → ConstantTerms}
    {context : PiCcsOutputContext
      Shape Commitment PublicX RowPoint ColumnPoint FoldDigest Sidecars}
    {left right : PiCcsOutputClaim
      Shape Commitment PublicX RowPoint ColumnPoint FoldDigest Sidecars
      YRing ConstantTerms YZcol}
    (hLeft : PiCcsOutputReconstructed constantTerms context left)
    (hRight : PiCcsOutputReconstructed constantTerms context right)
    (hMessage : piCcsOutputMessage left = piCcsOutputMessage right) :
    left = right := by
  rw [hLeft, hRight, hMessage]

/-- Pointwise batch form used by a fixed-size `Pi_CCS` output vector. -/
theorem piCcsOutputMessage_batch_injective
    {n : Nat}
    {Shape Commitment PublicX RowPoint ColumnPoint FoldDigest Sidecars
      YRing ConstantTerms YZcol : Type u}
    {constantTerms : YRing → ConstantTerms}
    {contexts : Fin n → PiCcsOutputContext
      Shape Commitment PublicX RowPoint ColumnPoint FoldDigest Sidecars}
    {left right : Fin n → PiCcsOutputClaim
      Shape Commitment PublicX RowPoint ColumnPoint FoldDigest Sidecars
      YRing ConstantTerms YZcol}
    (hLeft : ∀ i, PiCcsOutputReconstructed constantTerms (contexts i) (left i))
    (hRight : ∀ i, PiCcsOutputReconstructed constantTerms (contexts i) (right i))
    (hMessage : ∀ i, piCcsOutputMessage (left i) = piCcsOutputMessage (right i)) :
    left = right := by
  funext i
  exact piCcsOutputMessage_injective_on_reconstructed
    (hLeft i) (hRight i) (hMessage i)

end SuperNeo.FPrimeRecursiveVerifier

import NightstreamFPrime.Export.Stage1.ApplicationRetainedGeometry
import NightstreamFPrime.Export.Stage1.DirectPiRLCSamplerCompletePrefixPlan
import NightstreamFPrime.Export.Stage1.PilotOrdinaryDirectPlan
import NightstreamFPrime.Layout.ProductionRelation.CanonicalBlockAssignment
import NightstreamFPrime.Layout.ProductionRelation.PinFamilyPlan

/-!
Owns the four final relation rows that bind the recursive public instance to
the digest produced by the augmented step. The 270 public coordinates encode
`encHash(outputDigest)`; the prior-state public link remains retained private
advice in the low-norm relation.

This plan adds no source column or retained slot. It decodes each public digest
word with the exact HyperNova `decodeHashWord` weights and equates it to the
existing constrained pilot output-digest form.
-/

namespace NightstreamFPrime.Export.Stage1.RecursivePublicOutputPlan

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

def rowCount : Nat := 4

@[simp] theorem rowCount_eq : rowCount = 4 := by
  rfl

def publicFits
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth) :
    ProductionAssignment.publicWidth ≤ logicalWidth := by
  have complete := geometry.completeFits
  rw [ApplicationRetainedGeometry.completeLogicalWidth_eq] at complete
  norm_num [ProductionAssignment.publicWidth, ringDegree, publicRingColumns]
    at complete ⊢
  omega

def carrierPublicFits
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth) :
    ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth := by
  exact Nat.le_trans (publicFits geometry)
    (Phi81CarrierLayout.logicalWidth_le_carrierWidth logicalWidth)

def oneColumn
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth) :
    Fin logicalWidth :=
  ApplicationRetainedGeometry.oneColumn geometry

def pilotOrdinaryGeometry
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth) :
    PilotOrdinaryRetainedGeometry.Geometry application logicalWidth :=
  DirectPiDECPrefixPlan.pilotOrdinaryGeometry
    (DirectPiRLCSamplerCompletePrefixPlan.piDecGeometry
      (ApplicationRetainedGeometry.prefixGeometry geometry))

def publicColumn
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth)
    (column : Fin ProductionAssignment.publicWidth) : Fin logicalWidth :=
  CanonicalBlockAssignment.publicColumn (publicFits geometry) column

def publicBitIndex (word : Fin 4) (bit : Nat) :
    Fin ProductionAssignment.publicWidth :=
  ⟨1 + word.val * 64 + bit % 64, by
    have wordBound := word.isLt
    have bitBound := Nat.mod_lt bit (by decide : 0 < 64)
    rw [ProductionAssignment.publicWidth_eq]
    omega⟩

def publicInput
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth) :
    PaperAlgebra.PublicInput
      (logicalWidth := logicalWidth)
      (publicFits := carrierPublicFits geometry) :=
  fun column => assignment (publicColumn geometry column)

def bitForm
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth)
    (word : Fin 4) (bit : Nat) : SparseForm logicalWidth :=
  SparseForm.singleton (publicColumn geometry (publicBitIndex word bit))
    (Poseidon2.ofNat (2 ^ bit))

def publicWordForm
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth)
    (word : Fin 4) : SparseForm logicalWidth :=
  (List.range 64).foldl (fun form bit =>
    SparseForm.add form (bitForm geometry word bit)) .empty

private theorem foldlForms_eval
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth) (word : Fin 4)
    (bits : List Nat) (initial : SparseForm logicalWidth) :
    (bits.foldl (fun form bit =>
        SparseForm.add form (bitForm geometry word bit)) initial).eval assignment =
      bits.foldl (fun value bit =>
        value + Poseidon2.ofNat (2 ^ bit) *
          assignment (publicColumn geometry (publicBitIndex word bit)))
        (initial.eval assignment) := by
  induction bits generalizing initial with
  | nil => rfl
  | cons bit rest inductionHypothesis =>
      simp only [List.foldl_cons]
      rw [inductionHypothesis]
      simp [bitForm]

theorem publicWordForm_eval
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth) (word : Fin 4) :
    (publicWordForm geometry word).eval assignment =
      Lifecycle.decodeHashWord (publicInput geometry assignment) word := by
  unfold publicWordForm Lifecycle.decodeHashWord publicInput
  rw [foldlForms_eval]
  rw [SparseForm.empty_eval]
  have indexEq (bit : Nat) :
      publicBitIndex word bit =
        Lifecycle.digestBitIndexNat (logicalWidth := logicalWidth) word bit := by
    apply Fin.ext
    rfl
  simp_rw [indexEq]

def outputWordForm
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth)
    (word : Fin 4) : SparseForm logicalWidth :=
  (PilotOrdinaryDirectPlan.Location.outputDigest word).form
    (pilotOrdinaryGeometry geometry)

def difference
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth)
    (word : Fin rowCount) : SparseForm logicalWidth :=
  SparseForm.add (outputWordForm geometry word)
    (SparseForm.scale (-1) (publicWordForm geometry word))

def interface
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth) :
    PinFamilyPlan.Interface logicalWidth rowCount where
  oneColumn := oneColumn geometry
  value := difference geometry

theorem rowCount_le :
    rowCount ≤ 2 ^ Lifecycle.cubeVariables := by
  norm_num [rowCount, Lifecycle.cubeVariables]

def plan
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  PinFamilyPlan.plan (interface geometry) rowCount_le

@[simp] theorem plan_rowCount
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth) :
    (plan geometry).rowCount = 4 := by
  rfl

def Matches
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth) : Prop :=
  ∀ word, (outputWordForm geometry word).eval assignment =
    Lifecycle.decodeHashWord (publicInput geometry assignment) word

theorem rowsZero_iff_matches
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (oneColumn geometry) = 1) :
    (plan geometry).RowsZero assignment ↔ Matches geometry assignment := by
  have one' : assignment (interface geometry).oneColumn = 1 := by
    exact one
  rw [plan, PinFamilyPlan.planRowsZero_iff
    (interface geometry) rowCount_le assignment one']
  constructor
  · intro zeros word
    have zero := zeros word
    change (difference geometry word).eval assignment = 0 at zero
    unfold difference at zero
    rw [SparseForm.add_eval, SparseForm.scale_eval,
      publicWordForm_eval] at zero
    have subZero :
        (outputWordForm geometry word).eval assignment -
          Lifecycle.decodeHashWord (publicInput geometry assignment) word = 0 := by
      simpa [sub_eq_add_neg] using zero
    exact sub_eq_zero.mp subZero
  · intro matching word
    change (difference geometry word).eval assignment = 0
    unfold difference
    rw [SparseForm.add_eval, SparseForm.scale_eval,
      publicWordForm_eval, matching word]
    simp

theorem Matches.outputDigest_eq_decodeHash
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    {geometry : ApplicationRetainedGeometry.Geometry application logicalWidth}
    {assignment : Assignment F logicalWidth}
    (matching : Matches geometry assignment) :
    List.ofFn (fun word : Fin 4 =>
        (outputWordForm geometry word).eval assignment) =
      Lifecycle.decodeHash (publicInput geometry assignment) := by
  apply congrArg List.ofFn
  funext word
  exact matching word

theorem Matches.outputDigest_eq_of_encHash
    {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    {geometry : ApplicationRetainedGeometry.Geometry application logicalWidth}
    {assignment : Assignment F logicalWidth}
    (matching : Matches geometry assignment)
    (digest : Digest) (fixed : digest.length = 4)
    (publicEqual : publicInput geometry assignment =
      Lifecycle.encHash (publicFits := carrierPublicFits geometry) digest) :
    List.ofFn (fun word : Fin 4 =>
        (outputWordForm geometry word).eval assignment) = digest := by
  rw [matching.outputDigest_eq_decodeHash, publicEqual]
  exact Lifecycle.decodeHash_encHash digest fixed

end NightstreamFPrime.Export.Stage1.RecursivePublicOutputPlan

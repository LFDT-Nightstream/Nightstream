import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.ActiveSourceLayout
import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.Sis.Refinement

/-!
Target-side boundary from the active typed PiCCS source tree to one compact
seeded-Phi81 block.

Assurance tier: conditional implementation/R1CS correspondence.

Owns: conversion of canonical physical field values into the independent
active serializer; preservation of the three distinct source-authority
classes; and composition with the generic seeded-Phi81 block refinement.

Does not own: concrete columns or block metadata, canonical-word row
satisfaction, PiCCS output truth, the delayed `y_zcol` authority theorem,
public-seed expansion, Poseidon2, transcript placement, costs, necessity, or
row removal.

Emits constraints: no.

Authority boundary: the theorem requires verifier-shape, `y_ring`, and
`y_zcol` bindings independently. A holding SIS block or equal digest cannot
replace any of them. The Rust fixed-point audit must separately instantiate
the concrete 15-source/13-matrix column and block layout.

| Protocol → phase → family | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `pi_ccs.output_digest.source_fields` | physical source values equal the complete typed serializer | checked boundary | `decodedNatFields_eq_serialize` |
| `pi_ccs.output_digest.sis.primary.words` | canonical words encode those exact source values | checked R1CS premise | `Sis.Refinement.WordAgreement` |
| `pi_ccs.output_digest.sis.primary.map` | every output is the independent seeded linear map | derived | `outputs_eq_apply_of_bound` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout.SisBoundary

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout

/-- Canonical physical values embedded into the independent Goldilocks
carrier. -/
def fieldAssignment
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP) : Nat -> F :=
  fun column => ⟨assignment column, canonical column⟩

/-- Natural representatives read from the complete typed source-column tree. -/
def decodedNatFields
    {shape : SemanticShape}
    (assignment : Nat -> Nat)
    (column : SourceRole shape -> Nat) : List Nat :=
  (sourceRoles shape).map fun role => assignment (column role)

/-- The three explicit authority classes determine the complete active
serialization before any SIS compression occurs. -/
theorem decodedNatFields_eq_serialize
    {shape : SemanticShape}
    {assignment : Nat -> Nat}
    {canonical : forall column, assignment column < goldilocksP}
    {column : SourceRole shape -> Nat}
    {message : OutputMessage shape}
    (verifierShapeBound :
      BindingsHoldFor .verifierShape (fieldAssignment assignment canonical)
        column message)
    (yRingBound :
      BindingsHoldFor .yRingOutput (fieldAssignment assignment canonical)
        column message)
    (yZcolBound :
      BindingsHoldFor .yZcolOutput (fieldAssignment assignment canonical)
        column message) :
    decodedNatFields assignment column =
      (ActiveSemantics.serialize message).map Fin.val := by
  have decoded := decodedFields_eq_serialize
    (fieldAssignment assignment canonical) column message
    verifierShapeBound yRingBound yZcolBound
  have values := congrArg (List.map Fin.val) decoded
  simpa [decodedNatFields, decodedFields, fieldAssignment, List.map_map]
    using values

/-- A valid holding block whose canonical words consume the bound active
source tree materializes exactly the independent SIS map of the complete
typed PiCCS output serialization. -/
theorem outputs_eq_apply_of_bound
    {shape : SemanticShape}
    {block : SeededPhi81.Block}
    {assignment : Nat -> Nat}
    {canonical : forall column, assignment column < goldilocksP}
    {column : SourceRole shape -> Nat}
    {message : OutputMessage shape}
    (verifierShapeBound :
      BindingsHoldFor .verifierShape (fieldAssignment assignment canonical)
        column message)
    (yRingBound :
      BindingsHoldFor .yRingOutput (fieldAssignment assignment canonical)
        column message)
    (yZcolBound :
      BindingsHoldFor .yZcolOutput (fieldAssignment assignment canonical)
        column message)
    (valid : block.Valid)
    (holds : block.Holds assignment)
    (wordAgreement :
      Sis.Refinement.WordAgreement block
        (decodedNatFields assignment column) assignment) :
    block.outputColumns.map assignment =
      Sis.Semantics.apply (Sis.Refinement.mapOfBlock block)
        ((ActiveSemantics.serialize message).map Fin.val) := by
  have fields := decodedNatFields_eq_serialize
    verifierShapeBound yRingBound yZcolBound
  rw [fields] at wordAgreement
  exact Sis.Refinement.outputs_eq_apply valid holds wordAgreement

end Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout.SisBoundary

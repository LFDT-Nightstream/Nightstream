import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.ActiveSourceLayout.SisBoundary

/-! Public theorem-shape regression for the active PiCCS source-to-SIS
boundary. -/

namespace tests.PiCcsOutputActiveSisBoundary

open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout
open Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout.SisBoundary

example
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
      PiCcsOutputDigest.Sis.Refinement.WordAgreement block
        (decodedNatFields assignment column) assignment) :
    block.outputColumns.map assignment =
      PiCcsOutputDigest.Sis.Semantics.apply
        (PiCcsOutputDigest.Sis.Refinement.mapOfBlock block)
        ((PiCcsOutputDigest.ActiveSemantics.serialize message).map Fin.val) := by
  exact outputs_eq_apply_of_bound verifierShapeBound yRingBound yZcolBound
    valid holds wordAgreement

end tests.PiCcsOutputActiveSisBoundary

import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentityConcreteNifs
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.HonestCompleteness

/-!
Contract: executable honest completeness for the concrete
`PaddedRowIdentity` NIFS.

Owns: conversion of the direct logical selective-CCS source relation into
the concrete key's source relation, construction of one complete proof, and
successful execution of the concrete `Pi_CCS`--`Pi_RLC`--`Pi_DEC` verifier.

Does not own: a production matrix artifact, Rust, generated R1CS rows,
Poseidon2 collision security, Module-SIS hardness, or extraction soundness.

Assurance tier: model-level honest completeness. The theorem is for every
typed application matrix family with the selected production dimensions.
-/

set_option autoImplicit false
set_option maxHeartbeats 800000
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityConcreteCompleteness

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentity
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentitySoundness

namespace Algebra
export PaddedRowIdentityConcreteAlgebra
  (AjtaiKey Commitment PublicInput openingMaps)
end Algebra

namespace Concrete
export PaddedRowIdentityConcreteNifs (key)
end Concrete

/-- The direct logical selected source relation is exactly the source relation
consumed by the concrete noninteractive verifier. -/
theorem logicalSourceHolds_iff_sourceValid
    (statementId : F)
    (ajtaiKey : Algebra.AjtaiKey)
    (matrices : ApplicationMatrices)
    (running : Running K Algebra.Commitment Algebra.PublicInput shape)
    (fresh : Fresh Algebra.Commitment Algebra.PublicInput shape)
    (witness : OutputWitness shape assignmentColumns) :
    LogicalSourceHolds (Algebra.openingMaps ajtaiKey) productionGlobalParams
        matrices
        (Fin.addCases fresh.commitments running.commitments)
        (Fin.addCases fresh.publicInputs running.publicInputs)
        running.point
        (fun coordinate =>
          running.evaluations coordinate.running coordinate.matrix
            coordinate.coefficient)
        witness <->
      SourceValid (Concrete.key statementId ajtaiKey matrices)
        running fresh witness := by
  unfold SourceValid
  exact
    (sourceHolds_iff_logicalSourceHolds
      (Algebra.openingMaps ajtaiKey) productionGlobalParams matrices
      (Fin.addCases fresh.commitments running.commitments)
      (Fin.addCases fresh.publicInputs running.publicInputs)
      running.point
      (fun coordinate =>
        running.evaluations coordinate.running coordinate.matrix
          coordinate.coefficient)
      witness).symm

/-- Every direct logical source witness constructs one concrete proof which
the executable verifier accepts, together with an independently witnessed
paper transition. No cryptographic soundness assumption is needed for honest
completeness. -/
theorem logicalSource_exists_verifiedTransition
    (statementId : F)
    (ajtaiKey : Algebra.AjtaiKey)
    (matrices : ApplicationMatrices)
    (running : Running K Algebra.Commitment Algebra.PublicInput shape)
    (fresh : Fresh Algebra.Commitment Algebra.PublicInput shape)
    (witness : OutputWitness shape assignmentColumns)
    (source :
      LogicalSourceHolds (Algebra.openingMaps ajtaiKey) productionGlobalParams
        matrices
        (Fin.addCases fresh.commitments running.commitments)
        (Fin.addCases fresh.publicInputs running.publicInputs)
        running.point
        (fun coordinate =>
          running.evaluations coordinate.running coordinate.matrix
            coordinate.coefficient)
        witness) :
    exists proof : Proof K Algebra.Commitment shape 9,
    exists result : Running K Algebra.Commitment Algebra.PublicInput shape,
      verify (Concrete.key statementId ajtaiKey matrices)
          running fresh proof = some result /\
      Transition (Concrete.key statementId ajtaiKey matrices)
        running fresh result := by
  apply sourceValid_exists_verifiedTransition
    (Concrete.key statementId ajtaiKey matrices) running fresh witness
  exact
    (logicalSourceHolds_iff_sourceValid statementId ajtaiKey matrices
      running fresh
      witness).mp source

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityConcreteCompleteness

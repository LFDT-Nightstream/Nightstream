import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RewriteSourceSemantics

/-!
Focused regression for model-level closed rewrite-chain agreement.

The source assignment is constrained only by the independently executed
source definitions. In particular, this theorem application has no premise
asserting that the source assignment satisfies the compiler rewrite rows.
-/

namespace Nightstream.Tests.FPrimeSelectiveFixedPointPiCcsNcSourceRefinement

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.Decoder
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.Semantics
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveCompilerBridge
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RewriteChain
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RewriteSourceSemantics.ChainAgreement

#check ExactChainMatch
#check exactChainMatch_implies_sourceValue_eq_contributions
#check exactChainMatch_implies_sourceValue_eq_compiler_of_contributionsEqual

example
    {columns : Nat} {definitions : List Program.Definition}
    {steps : List (DecodedRewriteStep columns)}
    {output : DecodedLinearCombination columns}
    {sourceAssignment compilerAssignment : Nat → Nat}
    {compilerDerivedValue : Nat → F}
    (matching : ExactChainMatch definitions steps output)
    (definitionsHold : ∀ definition ∈ definitions,
      definition.Holds sourceAssignment)
    (chain : SourceChain none steps output)
    (compilerHolds : ∀ step ∈ steps,
      RewriteStepHolds compilerAssignment compilerDerivedValue step)
    (contributionsEqual : ∀ step ∈ steps,
      contribution sourceAssignment step =
        contribution compilerAssignment step) :
    linearCombinationValue output sourceAssignment =
      linearCombinationValue output compilerAssignment :=
  exactChainMatch_implies_sourceValue_eq_compiler_of_contributionsEqual
    matching definitionsHold chain compilerHolds contributionsEqual

end Nightstream.Tests.FPrimeSelectiveFixedPointPiCcsNcSourceRefinement

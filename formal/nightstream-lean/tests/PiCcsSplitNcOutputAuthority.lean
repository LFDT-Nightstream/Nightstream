import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement.OutputAuthority

/-!
Focused structural regressions for production terminal/output authority.
-/

set_option autoImplicit false

namespace NightstreamTests.PiCcsSplitNcOutputAuthority

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc
open Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement

universe uState

variable
  {shape : SemanticShape}
  {State : Type uState}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}

example
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits))
    (certificate : Certificate input)
    (paper : Semantics.Paper.Holds input.data) :
    OutputAuthority.Holds input certificate :=
  OutputAuthority.of_paper input certificate paper

example
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits))
    (certificate : Certificate input)
    (paper : Semantics.Paper.Holds input.data) :
    (Semantics.Paper.Holds input.data ∧
      OutputAuthority.Holds input certificate) ∨
      FeFailure input certificate ∨
      NcFailure input certificate ∨
      RegisteredDeviationObligation input certificate :=
  Or.inl ⟨paper, OutputAuthority.of_paper input certificate paper⟩

end NightstreamTests.PiCcsSplitNcOutputAuthority

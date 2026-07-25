import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.SemanticAdapter

/-!
Terminal and output authority for the production Split-NC prefix.

Assurance tier: model-level registered-deviation refinement.

Owns: exact FE and base-or-delayed NC terminal binding, the complete
verifier-materialized output product, and the acceptance-derived
paper-or-algebraic-failure partition.

Does not own: probability bounds, concrete field certificates, Fiat--Shamir,
commitment security, extraction, Rust, R1CS, costs, or rows.

Emits constraints: no.

| Stage path | Owned equation | Authority |
|---|---|---|
| `fprime.piccs.production.output_authority` | accepted production output is paper-bound or an exact FE/NC event occurs | derived |
-/

set_option autoImplicit false
set_option maxRecDepth 2048

namespace Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement.OutputAuthority

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial

universe uState

variable
  {shape : SemanticShape}
  {State : Type uState}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Every terminal and downstream output equality owned by the concrete
production prefix. The output message itself remains a verifier computation. -/
structure Holds
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits))
    (certificate : Certificate input) : Prop where
  yRingBound :
    ProductionPiCcs.YRingBound input.full input.data certificate.materialize
  packedYZcolBound :
    Polynomial.Nc.BlockLane.Terminal.PackedYZcolBoundAtBlock
      input.full.covers input.data
      (ProductionPiCcs.ncPoint input.full certificate.materialize).block
      certificate.materialize.piCcs.output
  feTerminal :
    Polynomial.Fe.terminalFromMessage input.full.profile
        (PublicInput.ofSources input.data) input.full.feCoins
        (ProductionPiCcs.fePoint input.full certificate.materialize)
        certificate.output =
      Polynomial.Fe.InitialSum.sumcheckPolynomial input.full.profile
        input.data input.full.feCoins
        (ProductionPiCcs.fePoint input.full
          certificate.materialize).coordinates
  ncTerminal :
    ProductionPiCcs.messageTerminal input.full certificate.materialize =
      ProductionPiCcs.rawPolynomial input.full input.data
        (ProductionPiCcs.ncPoint input.full
          certificate.materialize).coordinates
  piRlcOutputs :
    (derive input.full certificate.materialize).piCcsOutputs =
      PiCCS.honestOutputs (semantics input.full.key) input.full.input
        (InputAuthority.productAssignments input.data input.full.alignment)
        certificate.feExecution.challengePoint.row

private theorem ncTerminal
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits))
    (certificate : Certificate input) :
    ProductionPiCcs.messageTerminal input.full certificate.materialize =
      ProductionPiCcs.rawPolynomial input.full input.data
        (ProductionPiCcs.ncPoint input.full
          certificate.materialize).coordinates := by
  cases pendingEq : input.full.pending with
  | none =>
      simp only [ProductionPiCcs.messageTerminal,
        ProductionPiCcs.rawPolynomial, pendingEq]
      exact
        Polynomial.Nc.BlockLane.Terminal.terminal_eq_qAtPoint_of_bound
          input.full.covers input.data input.full.ncCoins
          (ProductionPiCcs.ncPoint input.full certificate.materialize)
          certificate.output certificate.output_bound.2 |>.trans
            (Polynomial.Nc.BlockLane.InitialSum.sumcheckPolynomial_coordinates_eq_qAtPoint
              input.full.covers input.data input.full.ncCoins
              (ProductionPiCcs.ncPoint input.full
                certificate.materialize)).symm
  | some pending =>
      simp only [ProductionPiCcs.messageTerminal,
        ProductionPiCcs.rawPolynomial, pendingEq]
      exact
        MessageTerminal.verifierTerminal_eq_sumcheckPolynomial_of_bound
          input.full.covers input.data input.full.ncCoins
          (ProductionProjection.productionWeights input.full)
          input.full.producerBeta input.full.batchWeight pending.oldBlock
          (ProductionPiCcs.ncPoint input.full certificate.materialize)
          certificate.output certificate.output_bound.2

/-- Paper truth plus canonical materialization supplies the complete terminal
and output authority predicate without an acceptance or no-failure premise. -/
theorem of_paper
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits))
    (certificate : Certificate input)
    (paper : Semantics.Paper.Holds input.data) :
    Holds input certificate := by
  refine {
    yRingBound := certificate.output_bound.1
    packedYZcolBound := certificate.output_bound.2
    feTerminal := ?_
    ncTerminal := ncTerminal input certificate
    piRlcOutputs := ?_
  }
  · exact
      Verifier.SumCheck.SemanticAdapter.feTerminalBinding input.full.profile
        input.data input.full.feCoins
        (ProductionPiCcs.fePoint input.full certificate.materialize)
        certificate.output certificate.output_bound.1
  · change
      OutputProduct.materialize publicRingColumns publicFits
          input.full.alignment input.full.input
          certificate.feExecution.challengePoint.row certificate.output =
        PiCCS.honestOutputs (semantics input.full.key) input.full.input
          (InputAuthority.productAssignments input.data input.full.alignment)
          certificate.feExecution.challengePoint.row
    simpa [semantics] using
      (Protocol.OutputRefinement.materializedOutputs_eq_honestOutputs_of_yRingEq
        publicRingColumns publicFits (commit input.full.key) input.data
        input.full.alignment input.full.input
        certificate.feExecution.challengePoint.row certificate.output
        production_norm_stages.1 paper input.sourceProduct_bound
        certificate.output_bound.1)

/-- The requested disjunctive production authority theorem. No caller supplies
the good branch or excludes algebraic events. -/
theorem accepted_implies_paper_and_authority_or_named_failure
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits))
    (certificate : Certificate input)
    (accepted : ProductionVerifierAccepts input certificate) :
    (Semantics.Paper.Holds input.data ∧ Holds input certificate) ∨
      FeFailure input certificate ∨
      NcFailure input certificate ∨
      RegisteredDeviationObligation input certificate := by
  rcases accepted_implies_paper_or_algebraic_failure noZeroDivisors input
      certificate accepted with paper | fe | nc
  · exact Or.inl ⟨paper, of_paper input certificate paper⟩
  · exact Or.inr (Or.inl fe)
  · exact Or.inr (Or.inr (Or.inl nc))

end Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement.OutputAuthority

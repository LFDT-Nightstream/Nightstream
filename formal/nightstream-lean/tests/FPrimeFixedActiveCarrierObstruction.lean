import Nightstream.Protocol.FPrime.Frozen.FixedActiveCarrierObstruction

/-!
Focused type-level regression for the conditional fixed-active carrier
necessity result. The statements compare complete 270-coordinate paper inputs
with hypothetical 257-coordinate instantiations of the carrier-polymorphic NIFS
and F' types; they do not assert that production selects the erased carrier.
-/

set_option autoImplicit false

namespace tests.FPrimeFixedActiveCarrierObstruction

open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.Protocol.FPrime.Frozen.FixedActiveCarrierObstruction

#check exactPaperVerifier
#check exactPaperVerifier_soundAndCompleteModulo
#check eraseRunning_zero_eq_tail
#check zeroPublicRunning_ne_tailMutatedRunning
#check no_exact_paperNifs_running_decoder
#check no_exact_fixedOne_fprime_decoder
#check no_exact_construction2_fprime_decoder

universe uExtension uCommitment uState uWitness uFresh uProof uKey

/-- If a paper NIFS running input is exposed only through the 257-coordinate
view, that view cannot be inverted on the complete domain. -/
example
    {dimensions : Dimensions}
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    (seed : ExactRunning dimensions Extension Commitment) :
    ¬ ∃ decode : ErasedRunning dimensions Extension Commitment ->
        ExactRunning dimensions Extension Commitment,
      ∀ running, decode (eraseRunning running) = running :=
  no_exact_paperNifs_running_decoder seed

/-- The same conditional lossiness holds for a hypothetical 257-wide
fixed-one F' instantiation. -/
example
    {dimensions : Dimensions}
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {State : Type uState}
    {Witness : Type uWitness}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    (seed : ExactFixedOneInput dimensions Extension Commitment
      State Witness Fresh Proof) :
    ¬ ∃ decode : ErasedFixedOneInput dimensions Extension Commitment
          State Witness Fresh Proof ->
        ExactFixedOneInput dimensions Extension Commitment
          State Witness Fresh Proof,
      ∀ input, decode (eraseFixedOneInput input) = input :=
  no_exact_fixedOne_fprime_decoder seed

/-- The same conditional lossiness holds for a hypothetical 257-wide generic
Construction-2 instantiation, including its prior counter. -/
example
    {dimensions : Dimensions}
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {Key : Type uKey}
    {State : Type uState}
    {Witness : Type uWitness}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    (seed : ExactConstruction2Input dimensions Extension Commitment
      Key State Witness Fresh Proof) :
    ¬ ∃ decode : ErasedConstruction2Input dimensions Extension Commitment
          Key State Witness Fresh Proof ->
        ExactConstruction2Input dimensions Extension Commitment
          Key State Witness Fresh Proof,
      ∀ input, decode (eraseConstruction2Input input) = input :=
  no_exact_construction2_fprime_decoder seed

end tests.FPrimeFixedActiveCarrierObstruction

import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.RewindableContinuation

/-!
Kernel obstruction to sampling multiple `Pi_RLC` base vectors under one fixed
realized NIFS key.

Source: SuperNeo Appendix D.5, where `(pp, s, u₁, st)` is fixed and the
complete challenge vector is the coordinate extractor's varying input.

Owns: the fixed-key fork carrier for one continuation; proof that two valid
programming receipts force identical base vectors; and the consequent
failure theorem for distinct sampled bases.

Does not own: an oracle-world carrier, a random-oracle distribution, a
forking probability theorem, event bounds, Poseidon2, Ajtai, Rust, R1CS,
artifacts, minimality, or costs.

Emits constraints: no.

`Key.piRlcResponse` is a realized function.  Once the key, public input, and
fixed prefix are held constant, its base vector is fixed.  Therefore a
non-degenerate challenge experiment must vary the oracle realization rather
than claim that every sample is aligned to one key.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.FPrime.Frozen.NonInteractiveFixedKeyObstruction

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
open Nightstream.SuperNeo.InteractiveReduction.CoordinateForking

universe uExtension uCommitment uPublicInput uScalar uState

/-- Put an arbitrary fork sample around one fixed key, public input, and
rewindable continuation. -/
def outcomeForSample
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (prover : RewindableProver key)
    (sample : ForkSample Scalar key.arity.total) :
    RewindableForkOutcome key where
  running := running
  fresh := fresh
  prover := prover
  sample := sample

/-- Under one realized key and one fixed prefix, two valid programming
receipts must name the same base challenge vector. -/
theorem programmingReceipts_force_same_base
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (prover : RewindableProver key)
    (left right : ForkSample Scalar key.arity.total)
    (leftReceipt : CoordinateProgrammingReceipt
      (outcomeForSample running fresh prover left).toAlignedForkOutcome)
    (rightReceipt : CoordinateProgrammingReceipt
      (outcomeForSample running fresh prover right).toAlignedForkOutcome) :
    left.base = right.base := by
  calc
    left.base =
        key.piRlcChallenges running fresh
          (prover.baseProof running fresh) :=
      leftReceipt.baseAligned
    _ = right.base := rightReceipt.baseAligned.symm

/-- Distinct base samples cannot both be programmed correctly under one
fixed realized key.  At least one sample therefore inhabits the exact named
multi-fork programming failure event. -/
theorem distinct_bases_force_programming_failure
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (prover : RewindableProver key)
    (left right : ForkSample Scalar key.arity.total)
    (different : left.base ≠ right.base) :
    MultiForkProgrammingFailure
        (outcomeForSample running fresh prover left).toAlignedForkOutcome ∨
      MultiForkProgrammingFailure
        (outcomeForSample running fresh prover right).toAlignedForkOutcome := by
  by_cases leftProgrammed : CoordinateProgrammingReceipt
      (outcomeForSample running fresh prover left).toAlignedForkOutcome
  · right
    intro rightProgrammed
    exact different
      (programmingReceipts_force_same_base running fresh prover left right
        leftProgrammed rightProgrammed)
  · exact Or.inl leftProgrammed

end Nightstream.Protocol.FPrime.Frozen.NonInteractiveFixedKeyObstruction

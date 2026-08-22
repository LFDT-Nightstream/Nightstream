import NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive.Verifier
import NightstreamFPrime.Spec.Folding.PiCCS.v1_1.Statement
import NightstreamFPrime.Spec.Folding.PiCCS.v1_1.EvalK
import NightstreamFPrime.Spec.Folding.PiCCS.v1_1.EvalA
import NightstreamFPrime.Spec.Folding.PiCCS.v1_1.FinalIdentity

/-!
Paper authority: SuperNeo v1.1, Section 7.3, complete `Pi_CCS` verifier.
Obligation: Replay the verifier transcript, bind separate `Eval_K` and
`Eval_A` inputs and outputs, and check the fixed-width SumCheck chain against
the exact v1.1 initial and terminal formulas.

Inputs:
- the one production NIFS key;
- running and fresh public claims;
- the sole typed prover proof.

Outputs:
- the canonical production PiCCS acceptance predicate.

Constraint groups:
- statement and prior-claim binding;
- verifier-derived transcript coins;
- fixed-width SumCheck chain;
- separate complete Pad and CCS-matrix output families;
- exact final joint identity.

Parent coverage:
- `Nifs.PaperNonInteractive.piCcsCheck`.

`Accepted` is an abbreviation of the production check. `Coverage` is only a
proof view used to audit that check. It is not a second verifier path and
emits no circuit constraints.
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.v1_1

open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources
open NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive

universe uExtension uCommitment uPublicInput uScalar uState

/-- The exact production PiCCS acceptance predicate. -/
abbrev Accepted
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound) : Prop :=
  piCcsCheck key running fresh proof = true

/-- Mechanical coverage view of the canonical production check. Every field
is either a definitional binding theorem or the one accepted chain. -/
structure Coverage
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound) : Prop where
  transcript :
    (key.piCcsExecution running fresh proof).coins =
      FiatShamir.derive key.oracle.transcript
        ({ priorState := key.publicInputState running fresh
           input := (key.statement running fresh).verifierInput key.lift } :
          ProtocolVerifier.Statement Extension State shape)
        ({ rounds := fun round => (proof.piCcsRounds round).toMessage } :
          FiatShamir.Certificate Extension shape)
  input_eval_K : ∀ coordinate : PadCoordinate shape,
    ((key.statement running fresh).verifierInput key.lift
      ).claimedPadCoefficient coordinate =
        (running.evaluations coordinate.running).pad coordinate.coefficient
  input_eval_A : ∀ coordinate : MatrixCoordinate shape,
    ((key.statement running fresh).verifierInput key.lift
      ).claimedMatrixCoefficient coordinate =
        (running.evaluations coordinate.running).matrix coordinate.matrix
          coordinate.coefficient
  output_eval_K : ∀ coordinate : PadCoordinate shape,
    (key.piCcsCertificate running fresh proof).output.padImage coordinate =
      proof.piCcsOutput.padCoordinate
        (runningSourceIndex coordinate.running) coordinate.coefficient
  output_eval_A : ∀ coordinate : MatrixCoordinate shape,
    (key.piCcsCertificate running fresh proof).output.matrixImage coordinate =
      proof.piCcsOutput.matrixCoordinate
        (runningSourceIndex coordinate.running) coordinate.matrix
          coordinate.coefficient
  chain :
    SumCheck.Finite.FixedPhase.Chain key.extensionOps.toOps
      (((key.statement running fresh).verifierInput key.lift).initial
        key.extensionOps
        (key.piCcsExecution running fresh proof).coins.gamma)
      (key.piCcsFixedCertificate running fresh proof).rounds
      (key.piCcsExecution running fresh proof).coins.roundPoint.coordinates
      (ProtocolPolynomial.terminalFromMessage key.extensionOps
        ((key.statement running fresh).verifierInput key.lift)
        (key.piCcsExecution running fresh proof).coins.alpha
        (key.piCcsExecution running fresh proof).coins.gamma
        (key.piCcsExecution running fresh proof).coins.roundPoint
        (key.piCcsCertificate running fresh proof).output)

/-- The proof-only coverage view is exactly equivalent to production
acceptance. No additional predicate can make the verifier accept. -/
theorem accepted_iff_coverage
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound) :
    Accepted key running fresh proof ↔ Coverage key running fresh proof := by
  constructor
  · intro accepted
    refine {
      transcript := key.piCcsExecution_coins_eq_derive running fresh proof
      input_eval_K := ?_
      input_eval_A := ?_
      output_eval_K := ?_
      output_eval_A := ?_
      chain := (piCcsCheck_eq_true_iff key running fresh proof).mp accepted
    }
    · intro coordinate
      rfl
    · intro coordinate
      rfl
    · intro coordinate
      rfl
    · intro coordinate
      rfl
  · intro coverage
    exact (piCcsCheck_eq_true_iff key running fresh proof).mpr coverage.chain

end NightstreamFPrime.Spec.Folding.PiCCS.v1_1

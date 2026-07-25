import Nightstream.SuperNeo.SumCheck.FixedPhase

/-!
Assignment-independent wire data and assignment-indexed semantic views for a
fixed-width SumCheck phase.

Assurance tier: model-level.

Owns: a ghost-free wire transcript, its claimed-chain acceptance predicate,
reconstruction of the generic symbolic `SumCheck.Instance` from an explicit
semantic polynomial, and acceptance/truth-path transport under exact terminal
binding.

Does not own: a protocol polynomial, assignment extraction, challenge
generation, root counting, Fiat--Shamir, Rust, R1CS, costs, or rows.

The wire carrier contains no `q`, `trueInitial`, or expected-round callback.
Those values enter only through `semanticInstance`, after an assignment-indexed
protocol polynomial has been fixed.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.SemanticView

universe uField

/-- Everything the verifier needs to replay one fixed-width claimed chain.
The terminal is verifier-computed but remains independent of the semantic
polynomial until a separate binding theorem identifies the two. -/
structure Wire (Field : Type uField) (degree : Nat) where
  initial : Field
  terminal : Field
  challenges : List Field
  certificate : FixedPhase.Certificate Field degree
  challengeSetSize : Nat

/-- Ghost-free claimed-chain acceptance. -/
def Accepted
    {Field : Type uField}
    {degree : Nat}
    (ops : Ops Field)
    (wire : Wire Field degree) : Prop :=
  FixedPhase.Chain ops wire.initial wire.certificate.rounds wire.challenges
    wire.terminal

/-- Recompute all semantic fields from one explicit polynomial while
preserving the verifier-visible initial, challenges, messages, and support
cardinality. -/
def semanticInstance
    {Field : Type uField}
    {degree : Nat}
    (ops : Ops Field)
    (q : List Field -> Field)
    (wire : Wire Field degree) :
    Nightstream.SuperNeo.SumCheck.Instance Field Field :=
  FixedPhase.symbolicInstance ops q degree wire.challengeSetSize wire.initial
    wire.challenges wire.certificate

/-- Exact terminal binding upgrades the wire chain to the semantic
fixed-phase acceptance relation. -/
theorem fixedAccepted_of_terminalBinding
    {Field : Type uField}
    {degree : Nat}
    (ops : Ops Field)
    (q : List Field -> Field)
    (wire : Wire Field degree)
    (accepted : Accepted ops wire)
    (terminalBinding : wire.terminal = q wire.challenges) :
    FixedPhase.Accepted ops q wire.initial wire.challenges wire.certificate := by
  unfold FixedPhase.Accepted
  rw [← terminalBinding]
  exact accepted

/-- Claimed-chain acceptance transports to the generic semantic instance
without accepting any caller-supplied ghost. -/
theorem accepted_implies_symbolicAccepted
    {Field : Type uField}
    {degree : Nat}
    (ops : Ops Field)
    (q : List Field -> Field)
    (wire : Wire Field degree)
    (accepted : Accepted ops wire)
    (terminalBinding : wire.terminal = q wire.challenges) :
    Nightstream.SuperNeo.SumCheck.Accepted ops.toSymbolic
      (semanticInstance ops q wire) := by
  exact FixedPhase.accepted_implies_symbolicAccepted ops q
    wire.challengeSetSize wire.initial wire.challenges wire.certificate
    (fixedAccepted_of_terminalBinding ops q wire accepted terminalBinding)

/-- The independently recomputed expected rounds form the generic truth path.
Only claimed-chain shape and exact terminal binding are consumed. -/
theorem accepted_implies_truthPath
    {Field : Type uField}
    {degree : Nat}
    (ops : Ops Field)
    (q : List Field -> Field)
    (wire : Wire Field degree)
    (accepted : Accepted ops wire)
    (terminalBinding : wire.terminal = q wire.challenges) :
    Nightstream.SuperNeo.SumCheck.TruthPath ops.toSymbolic
      (semanticInstance ops q wire) := by
  exact FixedPhase.symbolicTruthPath ops q wire.challengeSetSize wire.initial
    wire.challenges wire.certificate
    (fixedAccepted_of_terminalBinding ops q wire accepted terminalBinding)

/-- Claim truth in the semantic view is exactly equality between the
verifier-owned initial claim and the independently recomputed cube sum. -/
theorem claimTrue_iff
    {Field : Type uField}
    {degree : Nat}
    (ops : Ops Field)
    (q : List Field -> Field)
    (wire : Wire Field degree) :
    Nightstream.SuperNeo.SumCheck.Claim.True
        (semanticInstance ops q wire) ↔
      wire.initial =
        FixedPhase.semanticInitial ops q wire.challenges.length := by
  rfl

end Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.SemanticView

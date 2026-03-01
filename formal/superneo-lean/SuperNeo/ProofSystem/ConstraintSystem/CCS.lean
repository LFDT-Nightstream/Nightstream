namespace SuperNeo.ProofSystem.ConstraintSystem

/-- Paper-facing CCS relation object. -/
structure CCS where
  arity : Nat
  constraints : Nat

/-- Paper-facing CE relation object. -/
structure CE where
  arity : Nat
  normBound : Nat

/-- Paper-facing relaxed CE relation object. -/
structure CERelaxed where
  arity : Nat
  slackBound : Nat

/-- Promote CE into relaxed CE form. -/
def CERelaxed.ofCE (ce : CE) : CERelaxed :=
  { arity := ce.arity, slackBound := ce.normBound }

end SuperNeo.ProofSystem.ConstraintSystem

namespace SuperNeo.ProofSystem

/-- Public context type for the proof-system facade. -/
structure Context where
  securityParam : Nat

/-- Public claim type for the proof-system facade. -/
structure Claim where
  id : Nat

/-- Public witness type for the proof-system facade. -/
structure Witness where
  id : Nat

end SuperNeo.ProofSystem

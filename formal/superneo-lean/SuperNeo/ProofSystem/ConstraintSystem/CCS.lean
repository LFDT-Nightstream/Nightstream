import SuperNeo.Norm

namespace SuperNeo.ProofSystem.ConstraintSystem

universe u

/-- Abstract commitment map `L : R_F^n -> C` from Definitions 12–14. -/
abbrev CommitmentMap (Commitment : Type u) := Coeffs → Commitment

/-- Abstract projector `L_in` from Definition 13 / Definition 14. -/
abbrev InputProjector := Coeffs → Coeffs

/--
Paper-facing Section 7.1 structure object.

`constraintHolds` semantically represents the polynomial-side predicate
`f(\bar(M_1 z), ..., \bar(M_t z)) ∈ Z^{log m}` once the encoded matrix images
have been produced.
-/
structure CCSStructure where
  matrices : Array (Array Coeffs)
  imageFamily : Coeffs → Array Coeffs
  evaluationFamily : Coeffs → Coeffs → Array Coeffs
  constraintHolds : Array Coeffs → Prop

namespace CCSStructure

/-- Number of matrix constraints carried by the structure. -/
def arity (s : CCSStructure) : Nat :=
  s.matrices.size

end CCSStructure

/--
Global reduction parameters from Definition 14, restricted to the relation-level
data used by Section 7.
-/
structure GlobalParams (Commitment : Type u) where
  challengeSet : Array Coeffs
  commitMap : CommitmentMap Commitment
  inputProjector : InputProjector
  shape : CCSStructure

/-- Paper-facing CCS relation object from Definition 12. -/
structure CCS (Commitment : Type u) where
  normBound : Nat
  commitMap : CommitmentMap Commitment
  shape : CCSStructure

namespace CCS

/-- Arity of a CCS relation is inherited from its structure. -/
def arity {Commitment : Type u} (ccs : CCS Commitment) : Nat :=
  ccs.shape.arity

/-- Public CCS statement `(c, x)` from Definition 12. -/
structure Statement (Commitment : Type u) where
  commitment : Commitment
  publicInput : Coeffs

/-- Private CCS witness `w` from Definition 12. -/
structure Witness where
  privateInput : Coeffs

/-- Combined vector `z := [x, w]` from Definition 12. -/
def fullVector
  {Commitment : Type u}
  (stmt : Statement Commitment)
  (wit : Witness) : Coeffs :=
  stmt.publicInput ++ wit.privateInput

/-- Norm-bounded CCS membership predicate from Definition 12. -/
def Holds
  {Commitment : Type u}
  (ccs : CCS Commitment)
  (stmt : Statement Commitment)
  (wit : Witness) : Prop :=
    let z := fullVector stmt wit
  stmt.commitment = ccs.commitMap z ∧
    normInfCoeffs z < ccs.normBound ∧
    ccs.shape.constraintHolds (ccs.shape.imageFamily z)

theorem holds_iff
  {Commitment : Type u}
  {ccs : CCS Commitment}
  {stmt : Statement Commitment}
  {wit : Witness} :
  Holds ccs stmt wit ↔
    let z := fullVector stmt wit
    stmt.commitment = ccs.commitMap z ∧
      normInfCoeffs z < ccs.normBound ∧
      ccs.shape.constraintHolds (ccs.shape.imageFamily z) := by
  rfl

end CCS

/-- Paper-facing CE relation object from Definition 13. -/
structure CE (Commitment : Type u) where
  normBound : Nat
  commitMap : CommitmentMap Commitment
  inputProjector : InputProjector
  shape : CCSStructure

namespace CE

/-- Arity of a CE relation is inherited from its structure. -/
def arity {Commitment : Type u} (ce : CE Commitment) : Nat :=
  ce.shape.arity

/-- Public CE statement `(c, x, r, {y_j})` from Definition 13. -/
structure Statement (Commitment : Type u) where
  commitment : Commitment
  publicInput : Coeffs
  point : Coeffs
  evaluations : Array Coeffs

/-- Witness `z` from Definition 13. -/
structure Witness where
  assignment : Coeffs

/-- Norm-bounded CE membership predicate from Definition 13. -/
def Holds
  {Commitment : Type u}
  (ce : CE Commitment)
  (stmt : Statement Commitment)
  (wit : Witness) : Prop :=
  stmt.commitment = ce.commitMap wit.assignment ∧
    stmt.publicInput = ce.inputProjector wit.assignment ∧
    normInfCoeffs wit.assignment < ce.normBound ∧
    stmt.evaluations =
      ce.shape.evaluationFamily wit.assignment stmt.point

theorem holds_iff
  {Commitment : Type u}
  {ce : CE Commitment}
  {stmt : Statement Commitment}
  {wit : Witness} :
  Holds ce stmt wit ↔
    stmt.commitment = ce.commitMap wit.assignment ∧
      stmt.publicInput = ce.inputProjector wit.assignment ∧
      normInfCoeffs wit.assignment < ce.normBound ∧
      stmt.evaluations =
        ce.shape.evaluationFamily wit.assignment stmt.point := by
  rfl

end CE

/-- Relaxed CE carrier used by folding after random linear combination. -/
structure CERelaxed (Commitment : Type u) where
  normBound : Nat
  slackBound : Nat
  commitMap : CommitmentMap Commitment
  inputProjector : InputProjector
  shape : CCSStructure

namespace CERelaxed

/-- Arity of a relaxed CE relation is inherited from its structure. -/
def arity {Commitment : Type u} (ce : CERelaxed Commitment) : Nat :=
  ce.shape.arity

/-- Public relaxed-CE statement reuses the CE public tuple shape. -/
abbrev Statement (Commitment : Type u) := CE.Statement Commitment

/-- Relaxed witness extends `z` with an explicit slack vector. -/
structure Witness where
  assignment : Coeffs
  slack : Coeffs

/-- Relaxed CE membership predicate: CE conditions plus a slack bound. -/
def Holds
  {Commitment : Type u}
  (ce : CERelaxed Commitment)
  (stmt : Statement Commitment)
  (wit : Witness) : Prop :=
  stmt.commitment = ce.commitMap wit.assignment ∧
    stmt.publicInput = ce.inputProjector wit.assignment ∧
    normInfCoeffs wit.assignment < ce.normBound ∧
    stmt.evaluations =
      ce.shape.evaluationFamily wit.assignment stmt.point ∧
    normInfCoeffs wit.slack ≤ ce.slackBound

/-- Promote CE into relaxed CE form with the same base norm bound as slack bound. -/
def ofCE {Commitment : Type u} (ce : CE Commitment) : CERelaxed Commitment :=
  { normBound := ce.normBound
    slackBound := ce.normBound
    commitMap := ce.commitMap
    inputProjector := ce.inputProjector
    shape := ce.shape }

/--
Any CE witness induces a relaxed CE witness by taking zero slack.
-/
theorem holds_of_ce
  {Commitment : Type u}
  {ce : CE Commitment}
  {stmt : CE.Statement Commitment}
  {wit : CE.Witness}
  (h : CE.Holds ce stmt wit) :
  Holds (ofCE ce) stmt ⟨wit.assignment, #[]⟩ := by
  rcases h with ⟨hCommit, hInput, hNorm, hEval⟩
  refine ⟨hCommit, hInput, hNorm, hEval, ?_⟩
  simp [normInfCoeffs]

end CERelaxed

namespace GlobalParams

/-- Build the norm-bounded CCS relation for the chosen bound `b`. -/
def ccs
  {Commitment : Type u}
  (params : GlobalParams Commitment)
  (b : Nat) : CCS Commitment :=
  { normBound := b
    commitMap := params.commitMap
    shape := params.shape }

/-- Build the norm-bounded CE relation for the chosen bound `b`. -/
def ce
  {Commitment : Type u}
  (params : GlobalParams Commitment)
  (b : Nat) : CE Commitment :=
  { normBound := b
    commitMap := params.commitMap
    inputProjector := params.inputProjector
    shape := params.shape }

/-- Build the relaxed CE carrier for bounds `b ≤ B`. -/
def ceRelaxed
  {Commitment : Type u}
  (params : GlobalParams Commitment)
  (b B : Nat) : CERelaxed Commitment :=
  { normBound := b
    slackBound := B
    commitMap := params.commitMap
    inputProjector := params.inputProjector
    shape := params.shape }

end GlobalParams

/--
One coherent paper-facing Section 7.1 theorem instance over a shared Definition
14 parameter package.

This is the proof-system-level owner of the CCS/CE tuple data before any
specialization to a compact protocol context.
-/
structure Section71Objects (Commitment : Type u) where
  params : GlobalParams Commitment
  normBound : Nat
  ccsStatement : CCS.Statement Commitment
  ccsWitness : CCS.Witness
  ceStatement : CE.Statement Commitment
  ceWitness : CE.Witness
  sharedCommitment :
    ccsStatement.commitment = ceStatement.commitment
  sharedPublicInput :
    ccsStatement.publicInput = ceStatement.publicInput
  sharedAssignment :
    CCS.fullVector ccsStatement ccsWitness = ceWitness.assignment

namespace Section71Objects

/-- Canonical realized CCS carrier from the shared Definition-14 parameters. -/
def ccs
  {Commitment : Type u}
  (h : Section71Objects Commitment) :
  CCS Commitment :=
  GlobalParams.ccs h.params h.normBound

/-- Canonical realized CE carrier from the shared Definition-14 parameters. -/
def ce
  {Commitment : Type u}
  (h : Section71Objects Commitment) :
  CE Commitment :=
  GlobalParams.ce h.params h.normBound

/-- Recover that the realized CCS and CE statements share one commitment. -/
theorem sharedCommitment_eq
  {Commitment : Type u}
  (h : Section71Objects Commitment) :
  h.ccsStatement.commitment = h.ceStatement.commitment :=
  h.sharedCommitment

/-- Recover that the realized CCS and CE statements share one public input. -/
theorem sharedPublicInput_eq
  {Commitment : Type u}
  (h : Section71Objects Commitment) :
  h.ccsStatement.publicInput = h.ceStatement.publicInput :=
  h.sharedPublicInput

/-- Recover that the CE witness assignment is the CCS full vector `[x, w]`. -/
theorem ceAssignment_eq_fullVector
  {Commitment : Type u}
  (h : Section71Objects Commitment) :
  h.ceWitness.assignment = CCS.fullVector h.ccsStatement h.ccsWitness := by
  simpa using h.sharedAssignment.symm

end Section71Objects

/--
One concrete proof-system Section 7.1 theorem instance: coherent CCS/CE tuple
data together with membership proofs for both relations.
-/
structure Section71Instance (Commitment : Type u)
    extends Section71Objects Commitment where
  ccsHolds :
    CCS.Holds (GlobalParams.ccs params normBound) ccsStatement ccsWitness
  ceHolds :
    CE.Holds (GlobalParams.ce params normBound) ceStatement ceWitness

namespace Section71Instance

/-- Forget the holds proofs and recover the underlying Section 7.1 objects. -/
abbrev objects
  {Commitment : Type u}
  (h : Section71Instance Commitment) :
  Section71Objects Commitment :=
  h.toSection71Objects

/-- Canonical realized CCS carrier from the shared Definition-14 parameters. -/
def ccs
  {Commitment : Type u}
  (h : Section71Instance Commitment) :
  CCS Commitment :=
  h.objects.ccs

/-- Canonical realized CE carrier from the shared Definition-14 parameters. -/
def ce
  {Commitment : Type u}
  (h : Section71Instance Commitment) :
  CE Commitment :=
  h.objects.ce

/-- Recover that the realized CCS and CE statements share one commitment. -/
theorem sharedCommitment_eq
  {Commitment : Type u}
  (h : Section71Instance Commitment) :
  h.ccsStatement.commitment = h.ceStatement.commitment :=
  h.objects.sharedCommitment_eq

/-- Recover that the realized CCS and CE statements share one public input. -/
theorem sharedPublicInput_eq
  {Commitment : Type u}
  (h : Section71Instance Commitment) :
  h.ccsStatement.publicInput = h.ceStatement.publicInput :=
  h.objects.sharedPublicInput_eq

/-- Recover that the CE witness assignment is the CCS full vector `[x, w]`. -/
theorem ceAssignment_eq_fullVector
  {Commitment : Type u}
  (h : Section71Instance Commitment) :
  h.ceWitness.assignment = CCS.fullVector h.ccsStatement h.ccsWitness :=
  h.objects.ceAssignment_eq_fullVector

end Section71Instance

end SuperNeo.ProofSystem.ConstraintSystem

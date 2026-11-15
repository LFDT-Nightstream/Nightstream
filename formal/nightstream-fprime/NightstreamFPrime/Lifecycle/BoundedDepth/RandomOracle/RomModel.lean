import NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.LightFork
import NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.RefreshFork
import NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.UniquenessLink

/-! The random-oracle model of both bounded-depth folds as one oracle. A point is a light or a
refresh transcript address (`Point`), and the oracle is its light and refresh parts (`Oracle`).
The two transcripts share no sponge call prefix (`LightSchedule.challengeCalls_separated_refresh`),
so the two parts are independent.

The fold verifiers read their coins from the oracle (`folds`): a light fold reads
`LightFork.folds` on the light part, and a refresh fold reads `RefreshFork.folds` on the refresh
part. A ROM attempt is the oracle with a bounded-depth attempt (`RomAttempt`). It succeeds
(`Succeeds`) when `ForkLink.Succeeds` holds with the folds of its oracle: the walk's
`BoundedDepthReverse.Succeeds` with oracle coins in place of the sponge. It extracts
(`Extracted`) when `UniquenessLink.Extracted` holds: the walk's `Extracted`, which does not read
the oracle.

The oracle goes into the attempt and not into the circuit key. The fork games
(`LightFork.game`, `RefreshFork.game`), `PriorLink` and `ForkLink.SlotHolds` take a `CircuitKey`
and read the oracle as a separate argument; an oracle in the key would change the fold type that
`PriorLink` reads.

An adversary that queries both parts is a light adversary once the refresh part is fixed
(`restrictLight`), and a refresh adversary once the light part is fixed (`restrictRefresh`), with
the same query bound (`restrictLight_bound`, `restrictRefresh_bound`). A successful light attempt
succeeds in the light game of its key on the restricted adversary (`light_succeeds`), and
likewise for refresh (`refresh_succeeds`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.RomModel

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.RandomOracle
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.BoundedDepth
open NightstreamFPrime.Lifecycle.RandomOracle
open NightstreamFPrime.Lifecycle.RandomOracle.Test (Answer)
open NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.ForkLink (Attempt Folds)

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ RingCarrierLayout.carrierWidth logicalWidth}

section Types

variable (logicalWidth publicFits)

abbrev LightPoint := Test.Point (LightSchedule.schedule logicalWidth publicFits) 8

abbrev RefreshPoint := Test.Point (RefreshSchedule.schedule logicalWidth publicFits) 8

/-- A point of the bounded-depth random oracle: a light or a refresh transcript address. -/
abbrev Point := LightPoint logicalWidth publicFits ⊕ RefreshPoint logicalWidth publicFits

/-- The bounded-depth random oracle: its light part and its refresh part. -/
abbrev Oracle := LightFork.Oracle logicalWidth publicFits × RefreshFork.Oracle logicalWidth publicFits

/-- A ROM attempt: the oracle that the fold verifiers read, and a bounded-depth attempt. -/
abbrev RomAttempt := Oracle logicalWidth publicFits × Attempt logicalWidth publicFits

/-- The walk's extractor answer: a witness of the input claim and one of the fresh instance. -/
abbrev Answer :=
  PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits) ×
    PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)

end Types

/-- The oracle's answer at a light or a refresh point. -/
def answers (oracle : Oracle logicalWidth publicFits) : Point logicalWidth publicFits → Test.Answer :=
  Sum.elim oracle.1 oracle.2

section Restrict

variable {Output : Type}

theorem queryBound_mono {Address : Type} {computation : OracleComp Address Test.Answer Output}
    {bound : Nat} (bounded : computation.QueryBound bound) :
    ∀ {larger : Nat}, bound ≤ larger → computation.QueryBound larger := by
  induction bounded with
  | done output _ => exact fun {larger} _ => .done output larger
  | query point next bound _ induction =>
      intro larger within
      obtain ⟨smaller, rfl⟩ : ∃ smaller, larger = smaller + 1 := ⟨larger - 1, by omega⟩
      exact .query point next smaller fun answer => induction answer (by omega)

/-- The adversary as a light adversary: refresh queries read the fixed refresh part. -/
def restrictLight (refresh : RefreshFork.Oracle logicalWidth publicFits) :
    OracleComp (Point logicalWidth publicFits) Test.Answer Output →
      OracleComp (LightPoint logicalWidth publicFits) Test.Answer Output
  | .done output => .done output
  | .query point next =>
      match point with
      | .inl light => .query light fun answer => restrictLight refresh (next answer)
      | .inr other => restrictLight refresh (next (refresh other))

/-- The adversary as a refresh adversary: light queries read the fixed light part. -/
def restrictRefresh (light : LightFork.Oracle logicalWidth publicFits) :
    OracleComp (Point logicalWidth publicFits) Test.Answer Output →
      OracleComp (RefreshPoint logicalWidth publicFits) Test.Answer Output
  | .done output => .done output
  | .query point next =>
      match point with
      | .inl other => restrictRefresh light (next (light other))
      | .inr refresh => .query refresh fun answer => restrictRefresh light (next answer)

theorem run_restrictLight (oracle : Oracle logicalWidth publicFits)
    (computation : OracleComp (Point logicalWidth publicFits) Test.Answer Output) :
    (restrictLight oracle.2 computation).run oracle.1 = computation.run (answers oracle) := by
  induction computation with
  | done _ => rfl
  | query point next induction =>
      cases point with
      | inl light => exact induction (oracle.1 light)
      | inr other => exact induction (oracle.2 other)

theorem run_restrictRefresh (oracle : Oracle logicalWidth publicFits)
    (computation : OracleComp (Point logicalWidth publicFits) Test.Answer Output) :
    (restrictRefresh oracle.1 computation).run oracle.2 = computation.run (answers oracle) := by
  induction computation with
  | done _ => rfl
  | query point next induction =>
      cases point with
      | inl other => exact induction (oracle.1 other)
      | inr refresh => exact induction (oracle.2 refresh)

theorem restrictLight_bound {computation : OracleComp (Point logicalWidth publicFits) Test.Answer Output}
    {bound : Nat} (bounded : computation.QueryBound bound)
    (refresh : RefreshFork.Oracle logicalWidth publicFits) :
    (restrictLight refresh computation).QueryBound bound := by
  induction bounded with
  | done output bound => exact .done output bound
  | query point next bound _ induction =>
      cases point with
      | inl light => exact .query light _ bound fun answer => induction answer
      | inr other => exact queryBound_mono (induction (refresh other)) (Nat.le_succ bound)

theorem restrictRefresh_bound
    {computation : OracleComp (Point logicalWidth publicFits) Test.Answer Output}
    {bound : Nat} (bounded : computation.QueryBound bound)
    (light : LightFork.Oracle logicalWidth publicFits) :
    (restrictRefresh light computation).QueryBound bound := by
  induction bounded with
  | done output bound => exact .done output bound
  | query point next bound _ induction =>
      cases point with
      | inl other => exact queryBound_mono (induction (light other)) (Nat.le_succ bound)
      | inr refresh => exact .query refresh _ bound fun answer => induction answer

end Restrict

section Events

variable (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))

/-- The bounded-depth fold verifiers in the random-oracle model: a light fold reads the light part
of the oracle, and a refresh fold reads the refresh part. -/
noncomputable def folds (oracle : Oracle logicalWidth publicFits) : Folds logicalWidth publicFits
  | .light => LightFork.folds ajtai oracle.1 .light
  | .refresh => RefreshFork.folds ajtai oracle.2 .refresh

/-- The NIFS success event in the random-oracle model: `ForkLink.Succeeds` with the folds of the
attempt's oracle. -/
def Succeeds (attempt : RomAttempt logicalWidth publicFits) : Prop :=
  ForkLink.Succeeds ajtai (folds ajtai attempt.1) attempt.2

/-- The answer extracts the attempt (`UniquenessLink.Extracted`, the walk's `Extracted`). -/
def Extracted (attempt : RomAttempt logicalWidth publicFits) (answer : Answer logicalWidth publicFits) :
    Prop :=
  UniquenessLink.Extracted ajtai attempt.2 answer

/-- The success event reads the fold verifiers only at the attempt's step. -/
theorem succeeds_congr {folds folds' : Folds logicalWidth publicFits}
    {attempt : Attempt logicalWidth publicFits}
    (same : folds (Spec.HyperNova.Construction2.BoundedDepth.circuitStep attempt.link.fold.circuit) =
      folds' (Spec.HyperNova.Construction2.BoundedDepth.circuitStep attempt.link.fold.circuit)) :
    ForkLink.Succeeds ajtai folds attempt ↔ ForkLink.Succeeds ajtai folds' attempt := by
  have verifierEq : Spec.HyperNova.Construction2.BoundedDepth.verifier Folding.Schedule.productionLightFoldCap folds
        attempt.link.fold.circuit =
      Spec.HyperNova.Construction2.BoundedDepth.verifier Folding.Schedule.productionLightFoldCap folds'
        attempt.link.fold.circuit := by
    unfold Spec.HyperNova.Construction2.BoundedDepth.verifier
    rw [same]
  unfold ForkLink.Succeeds BoundedDepthReverse.Succeeds
  rw [verifierEq]

/-- A successful light attempt succeeds in the light game of its key, on the adversary
restricted to the light part. -/
theorem light_succeeds {oracle : Oracle logicalWidth publicFits}
    {computation : OracleComp (Point logicalWidth publicFits) Test.Answer
      (Attempt logicalWidth publicFits)}
    (succeeds : Succeeds ajtai (oracle, computation.run (answers oracle)))
    (light : Spec.HyperNova.Construction2.BoundedDepth.circuitStep (computation.run (answers oracle)).link.fold.circuit =
      .light) :
    Fork.Succeeds (LightFork.game ajtai (computation.run (answers oracle)).link.fold.key)
      (restrictLight oracle.2 computation) oracle.1 := by
  unfold Fork.Succeeds
  rw [run_restrictLight]
  refine ⟨rfl, (succeeds_congr ajtai ?_).mp succeeds⟩
  rw [light]
  rfl

/-- A successful refresh attempt succeeds in the refresh game of its key, on the adversary
restricted to the refresh part. -/
theorem refresh_succeeds {oracle : Oracle logicalWidth publicFits}
    {computation : OracleComp (Point logicalWidth publicFits) Test.Answer
      (Attempt logicalWidth publicFits)}
    (succeeds : Succeeds ajtai (oracle, computation.run (answers oracle)))
    (refresh : Spec.HyperNova.Construction2.BoundedDepth.circuitStep (computation.run (answers oracle)).link.fold.circuit =
      .refresh) :
    Fork.Succeeds (RefreshFork.game ajtai (computation.run (answers oracle)).link.fold.key)
      (restrictRefresh oracle.1 computation) oracle.2 := by
  unfold Fork.Succeeds
  rw [run_restrictRefresh]
  refine ⟨rfl, (succeeds_congr ajtai ?_).mp succeeds⟩
  rw [refresh]
  rfl

end Events

end NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.RomModel

import Nightstream.Protocol.FPrime.CanonicalTerminalVerifier

/-!
Payload evaluator specialized to the one-slot Construction-2 profile.

Assurance tier: model-level.

Owns: the payload-minimal recursive data, direct erased-payload evaluation,
derivation of the sole selected slot and one-based counter, and extensional
equality with the generic payload transition.

Does not own: concrete SuperNeo relations, Rust, R1CS, lowering, commitment
security, or costs.

Emits constraints: no.

The recursive evaluator retains exactly the prior public-link check, the sole
running-relation check, and the selected fresh-relation check. It carries no
prover-selected program counter and performs no finite dispatch loop. This
module does not own the exact outer proof envelope. Callers must wrap this
payload with `OuterTerminalProof.bottom` or `.recursive`.
-/

namespace Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne

open Nightstream.HyperNova.Construction2.Paper

universe uKey uDigest uState uWitness uRunning uRunningWitness uFresh
  uFreshWitness uProof uEncoded

/-- The sole slot in the fixed-one profile. -/
def selected : Fin 1 := ⟨0, by decide⟩

@[simp] theorem selected_val : selected.val = 0 := rfl

/-- Every typed fixed-one slot is the sole selected slot. -/
theorem fin_eq_selected (slot : Fin 1) : slot = selected := by
  exact Subsingleton.elim slot selected

/-- The one-based fixed-one program counter is exactly one. -/
@[simp] theorem oneBased_selected : oneBased selected = 1 := rfl

/-- The derived fixed-one counter satisfies the paper range predicate. -/
theorem selected_counter_in_range : InRange 1 (oneBased selected) := by
  simp [InRange]

/-- Any checked fixed-one counter selects the sole slot. -/
@[simp] theorem selectedIndex_eq_selected
    {pc : Nat}
    (valid : InRange 1 pc) :
    selectedIndex valid = selected := by
  exact fin_eq_selected (selectedIndex valid)

/-- Payload-minimal recursive terminal data for the fixed-one profile. The
program counter and one-element function carriers are verifier-computed. -/
structure Proof
    (Running : Type uRunning)
    (RunningWitness : Type uRunningWitness)
    (Fresh : Type uFresh)
    (FreshWitness : Type uFreshWitness) where
  running : Running
  runningWitness : RunningWitness
  fresh : Fresh
  freshWitness : FreshWitness

namespace Proof

/-- Reconstruct the generic paper carrier from the fixed-one payload. -/
def toGeneric
    {Running : Type uRunning}
    {RunningWitness : Type uRunningWitness}
    {Fresh : Type uFresh}
    {FreshWitness : Type uFreshWitness}
    (proof : Proof Running RunningWitness Fresh FreshWitness) :
    TerminalProof Running RunningWitness Fresh FreshWitness 1 where
  running := fun _ => proof.running
  runningWitness := fun _ => proof.runningWitness
  fresh := proof.fresh
  freshWitness := proof.freshWitness
  pc := oneBased selected

/-- Forget the generic proof fields derived by the fixed-one profile. -/
def erase
    {Running : Type uRunning}
    {RunningWitness : Type uRunningWitness}
    {Fresh : Type uFresh}
    {FreshWitness : Type uFreshWitness}
    (proof : TerminalProof Running RunningWitness Fresh FreshWitness 1) :
    Proof Running RunningWitness Fresh FreshWitness where
  running := proof.running selected
  runningWitness := proof.runningWitness selected
  fresh := proof.fresh
  freshWitness := proof.freshWitness

@[simp] theorem erase_toGeneric
    {Running : Type uRunning}
    {RunningWitness : Type uRunningWitness}
    {Fresh : Type uFresh}
    {FreshWitness : Type uFreshWitness}
    (proof : Proof Running RunningWitness Fresh FreshWitness) :
    erase proof.toGeneric = proof := by
  rfl

/-- A generic one-slot terminal proof round-trips whenever its paper counter
passes the fixed-one range check.  Its function-valued fields are necessarily
constant because `Fin 1` has only the selected slot. -/
theorem toGeneric_erase_of_inRange
    {Running : Type uRunning}
    {RunningWitness : Type uRunningWitness}
    {Fresh : Type uFresh}
    {FreshWitness : Type uFreshWitness}
    (proof : TerminalProof Running RunningWitness Fresh FreshWitness 1)
    (valid : InRange 1 proof.pc) :
    (erase proof).toGeneric = proof := by
  have counter : proof.pc = 1 :=
    Nat.le_antisymm valid.2 valid.1
  cases proof with
  | mk running runningWitness fresh freshWitness pc =>
      dsimp only at counter
      subst pc
      dsimp [erase, toGeneric]
      congr 1
      · funext slot
        rw [fin_eq_selected slot]
      · funext slot
        rw [fin_eq_selected slot]

end Proof

/-- The sole running-relation check.  This is the fixed-one specialization of
the generic finite conjunction, not an additional obligation. -/
def runningAccepted
    {Key : Type uKey}
    {Running : Type uRunning}
    {RunningWitness : Type uRunningWitness}
    {Fresh : Type uFresh}
    {FreshWitness : Type uFreshWitness}
    {relations : TerminalRelations Key Running RunningWitness Fresh FreshWitness 1}
    (checks : RelationChecks relations)
    (setupKeys : Fin 1 -> Key)
    (proof : Proof Running RunningWitness Fresh FreshWitness) : Bool :=
  checks.runningCheck selected (setupKeys selected)
    proof.running proof.runningWitness

/-- The generic finite running conjunction is exactly the sole direct check
on a canonical fixed-one proof. -/
theorem allRunningAccepted_eq_runningAccepted
    {Key : Type uKey}
    {Running : Type uRunning}
    {RunningWitness : Type uRunningWitness}
    {Fresh : Type uFresh}
    {FreshWitness : Type uFreshWitness}
    {relations : TerminalRelations Key Running RunningWitness Fresh FreshWitness 1}
    (checks : RelationChecks relations)
    (setupKeys : Fin 1 -> Key)
    (proof : Proof Running RunningWitness Fresh FreshWitness) :
    CanonicalTerminalVerifier.allRunningAccepted checks setupKeys
        proof.toGeneric =
      runningAccepted checks setupKeys proof := by
  apply Bool.eq_iff_iff.mpr
  unfold CanonicalTerminalVerifier.allRunningAccepted
  rw [CanonicalTerminalVerifier.finRange_all_eq_true_iff]
  constructor
  · intro accepted
    simpa [runningAccepted, Proof.toGeneric] using accepted selected
  · intro accepted slot
    have slotEq : slot = selected := fin_eq_selected slot
    subst slot
    simpa [runningAccepted, Proof.toGeneric] using accepted

/-- Direct erased-payload terminal evaluation. Iteration zero checks only the
endpoint. A positive iteration checks the prior link, the sole running
relation, and the selected fresh relation; it performs no NIFS call. -/
def eval
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {RunningWitness : Type uRunningWitness}
    {Fresh : Type uFresh}
    {FreshWitness : Type uFreshWitness}
    {NifsProof : Type uProof}
    {Encoded : Type uEncoded}
    [DecidableEq State]
    [DecidableEq Encoded]
    (setup : Setup Key Running Fresh NifsProof 1)
    (machine : Machine Key Digest State Witness Running Fresh Encoded 1)
    (relations : TerminalRelations Key Running RunningWitness Fresh FreshWitness 1)
    (checks : RelationChecks relations)
    (statement : TerminalStatement State)
    (proof : Proof Running RunningWitness Fresh FreshWitness) : Bool :=
  if statement.iteration = 0 then
    decide (statement.zi = statement.z0)
  else
    (decide (machine.freshPublic proof.fresh =
      machine.encodeInstance (machine.hash {
        verifierKeys := setup.verifierKeys
        iteration := statement.iteration
        z0 := statement.z0
        current := statement.zi
        running := fun _ => proof.running
        pc := oneBased selected
      })) && runningAccepted checks setup.verifierKeys proof) &&
    checks.freshCheck selected (setup.verifierKeys selected)
      proof.fresh proof.freshWitness

/-- Computed acceptance of the payload-minimal terminal verifier. -/
def Accepts
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {RunningWitness : Type uRunningWitness}
    {Fresh : Type uFresh}
    {FreshWitness : Type uFreshWitness}
    {NifsProof : Type uProof}
    {Encoded : Type uEncoded}
    [DecidableEq State]
    [DecidableEq Encoded]
    (setup : Setup Key Running Fresh NifsProof 1)
    (machine : Machine Key Digest State Witness Running Fresh Encoded 1)
    (relations : TerminalRelations Key Running RunningWitness Fresh FreshWitness 1)
    (checks : RelationChecks relations)
    (statement : TerminalStatement State)
    (proof : Proof Running RunningWitness Fresh FreshWitness) : Prop :=
  eval setup machine relations checks statement proof = true

/-- The direct fixed-one evaluator is exactly the generic evaluator after
reconstructing only the fields fixed by the profile. -/
theorem eval_eq_generic
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {RunningWitness : Type uRunningWitness}
    {Fresh : Type uFresh}
    {FreshWitness : Type uFreshWitness}
    {NifsProof : Type uProof}
    {Encoded : Type uEncoded}
    [DecidableEq State]
    [DecidableEq Encoded]
    (setup : Setup Key Running Fresh NifsProof 1)
    (machine : Machine Key Digest State Witness Running Fresh Encoded 1)
    (relations : TerminalRelations Key Running RunningWitness Fresh FreshWitness 1)
    (checks : RelationChecks relations)
    (statement : TerminalStatement State)
    (proof : Proof Running RunningWitness Fresh FreshWitness) :
    eval setup machine relations checks statement proof =
      CanonicalTerminalVerifier.eval setup machine relations checks statement
        proof.toGeneric := by
  unfold eval CanonicalTerminalVerifier.eval
  by_cases iterationZero : statement.iteration = 0
  · simp [iterationZero]
  · rw [if_neg iterationZero, if_neg iterationZero]
    rw [allRunningAccepted_eq_runningAccepted]
    simp only [Proof.toGeneric]
    rw [dif_pos selected_counter_in_range]
    simp only [selectedIndex_eq_selected, oneBased_selected]
    apply Bool.eq_iff_iff.mpr
    simp

/-- Acceptance agrees exactly with the generic executable checker on the
canonical fixed-one carrier. -/
theorem accepts_iff_generic
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {RunningWitness : Type uRunningWitness}
    {Fresh : Type uFresh}
    {FreshWitness : Type uFreshWitness}
    {NifsProof : Type uProof}
    {Encoded : Type uEncoded}
    [DecidableEq State]
    [DecidableEq Encoded]
    (setup : Setup Key Running Fresh NifsProof 1)
    (machine : Machine Key Digest State Witness Running Fresh Encoded 1)
    (relations : TerminalRelations Key Running RunningWitness Fresh FreshWitness 1)
    (checks : RelationChecks relations)
    (statement : TerminalStatement State)
    (proof : Proof Running RunningWitness Fresh FreshWitness) :
    Accepts setup machine relations checks statement proof <->
      CanonicalTerminalVerifier.eval setup machine relations checks statement
        proof.toGeneric = true := by
  unfold Accepts
  rw [eval_eq_generic]

/-- The payload-minimal fixed-one checker is extensionally equal to the
generic erased-payload terminal transition. -/
theorem accepts_iff_transition
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {RunningWitness : Type uRunningWitness}
    {Fresh : Type uFresh}
    {FreshWitness : Type uFreshWitness}
    {NifsProof : Type uProof}
    {Encoded : Type uEncoded}
    [DecidableEq State]
    [DecidableEq Encoded]
    (setup : Setup Key Running Fresh NifsProof 1)
    (machine : Machine Key Digest State Witness Running Fresh Encoded 1)
    (relations : TerminalRelations Key Running RunningWitness Fresh FreshWitness 1)
    (checks : RelationChecks relations)
    (statement : TerminalStatement State)
    (proof : Proof Running RunningWitness Fresh FreshWitness) :
    Accepts setup machine relations checks statement proof <->
      TerminalTransition setup machine relations statement proof.toGeneric := by
  rw [accepts_iff_generic]
  exact CanonicalTerminalVerifier.eval_eq_true_iff_transition
    setup machine relations checks statement proof.toGeneric

end Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne

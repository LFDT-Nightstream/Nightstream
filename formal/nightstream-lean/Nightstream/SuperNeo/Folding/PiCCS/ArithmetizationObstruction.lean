import Nightstream.SuperNeo.Folding.PiCCS

/-!
Exact model-level obstruction at the generic `PiCCS.Attempt` interface.

Owns: one finite countermodel in which the verifier accepts and all
assignment-dependent payload, norm, and ambient-output obligations hold, but
the fixed attempt's prover-independent FE initial value is incompatible with
those obligations.

Does not own: a claim that `PiCCS.Attempt` must be redesigned, a production
Split-NC adapter, Fiat--Shamir, Rust, R1CS, costs, or rows.

The countermodel isolates semantic-ghost opacity. `SumCheck.Accepted` follows
the claimed chain and cannot inspect `trueInitial`; `PiCCS.Arithmetization`
must additionally turn true payloads into equality with that hidden value.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.PiCCS.ArithmetizationObstruction

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding

/-- Small verifier-owned parameters with a genuine positive fresh batch. -/
def params : GlobalParams where
  q := 97
  b := 3
  k := 2
  maxFresh := 1
  expansionT := 1
  rlc_bound := by decide

/-- Bootstrap avoids introducing irrelevant running inputs. -/
def arity : BatchArity params :=
  BatchArity.bootstrap params 1 (by decide) (by decide)

/-- Every independent relation predicate is true in the countermodel. -/
def semantics : RelationSemantics Unit Unit Unit Unit Unit Unit where
  commit := fun _ => ()
  projectPublicInput := fun _ => ()
  normBounded := fun _ _ => True
  ccsSatisfied := fun _ _ => True
  evaluationPointValid := fun _ _ => True
  evaluations := fun _ _ _ => #[]

def sourceStatement : CCS.Instance Unit Unit Unit where
  constraintSystem := ()
  commitment := ()
  publicInput := ()
  stage := .fresh

def outputStatement : CE.Instance Unit Unit Unit Unit Unit where
  constraintSystem := ()
  commitment := ()
  publicInput := ()
  point := ()
  evaluations := #[]
  stage := .fresh

def inputs : PiCCS.InputProduct Unit Unit Unit Unit Unit params arity where
  fresh := fun _ => sourceStatement
  running := fun index => Fin.elim0 index

def ops : SumCheck.Ops Nat Nat where
  zero := 0
  one := 1
  add := Nat.add

/-- The accepted claimed chain is `0 = 0`, while the hidden true initial
value is deliberately incompatible with it. -/
def incompatibleFe : SumCheck.Instance Nat Nat where
  claimedInitial := 0
  trueInitial := 1
  terminal := 0
  rounds := []
  maxDegree := 0
  challengeSetSize := 1

def trivialNc : SumCheck.Instance Nat Nat where
  claimedInitial := 0
  trueInitial := 0
  terminal := 0
  rounds := []
  maxDegree := 0
  challengeSetSize := 1

def attempt :
    PiCCS.Attempt Unit Unit Unit Unit Unit Nat Nat params arity where
  inputs := inputs
  outputs := fun _ => outputStatement
  fe := incompatibleFe
  nc := trivialNc

def assignments : Fin arity.total -> Unit := fun _ => ()

private theorem sourceFresh (index : Fin arity.total) :
    (inputs.source index).stage = .fresh := by
  exact inputs.sourceCases (fun source => source.stage = .fresh)
    (fun _ => rfl) (fun empty => Fin.elim0 empty) index

/-- The generic verifier accepts the countermodel's exact fixed attempt. -/
theorem accepted : PiCCS.Accepted ops attempt := by
  refine ⟨{
    sourceFresh := sourceFresh
    outputFresh := fun _ => rfl
    sameStructure := fun _ => rfl
    sameCommitment := fun _ => rfl
    samePublicInput := fun _ => rfl
    sharedOutputPoint := fun _ _ => rfl
  }, ?_, ?_⟩
  · simp [SumCheck.Accepted, SumCheck.Chain, attempt, incompatibleFe]
  · simp [SumCheck.Accepted, SumCheck.Chain, attempt, trivialNc]

/-- Every independent FE payload obligation is true. -/
theorem payloadsHold : PiCCS.PayloadsHold semantics attempt assignments := by
  intro index
  exact inputs.sourceCases
    (fun source => PiCCS.Source.PayloadTruth semantics source ())
    (fun _ => trivial) (fun empty => Fin.elim0 empty) index

/-- Every independent norm obligation is true. -/
theorem normsHold : PiCCS.NormsHold semantics params assignments := by
  intro _
  trivial

/-- The same assignments open every accepted output at the ambient bound. -/
theorem ambientOutputsHold :
    PiCCS.AmbientOutputsHold semantics params attempt assignments := by
  intro _
  exact ⟨⟨rfl, rfl, trivial⟩, trivial, rfl⟩

/-- A fixed `PiCCS.Attempt` may contain a semantic initial value incompatible
with true payloads, even though its claimed chain is accepted. -/
theorem not_arithmetization :
    ¬ PiCCS.Arithmetization semantics params ops attempt assignments := by
  intro bridge
  have claimTrue := bridge.feClaimTrue_of_payloads payloadsHold
  change (0 : Nat) = 1 at claimTrue
  exact Nat.zero_ne_one claimTrue

/-- Exact requested obstruction: acceptance and every independent
assignment-side premise can hold while the fixed-attempt arithmetization
bridge is false. -/
theorem accepted_payloads_norms_ambient_without_arithmetization :
    ∃ (candidate :
        PiCCS.Attempt Unit Unit Unit Unit Unit Nat Nat params arity)
      (openings : Fin arity.total -> Unit),
      PiCCS.Accepted ops candidate ∧
      PiCCS.PayloadsHold semantics candidate openings ∧
      PiCCS.NormsHold semantics params openings ∧
      PiCCS.AmbientOutputsHold semantics params candidate openings ∧
      ¬ PiCCS.Arithmetization semantics params ops candidate openings := by
  exact ⟨attempt, assignments, accepted, payloadsHold, normsHold,
    ambientOutputsHold, not_arithmetization⟩

end Nightstream.SuperNeo.Folding.PiCCS.ArithmetizationObstruction

import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Poseidon23HashPriorRecipe
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Poseidon23HashNextRecipe
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.RunningCheckRecipe
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.FreshCheckRecipe

/-!
Contract: the application-supplied certification boundary for the four
application-dependent calls closed by canonical Phases 3 and 4.

HyperNova Construction 2 supplies the application circuits and their
encodings to setup.  Consequently fixed-one/plain/carrier-width parameters do
not determine these recipes.  This structure packages executable row programs
whose `CallRecipe` fields kernel-check their exact frozen call semantics.

Owns: one codec profile, total `hashPrior` and `hashNext` recipes, and
independent `runningCheck` and `freshCheck` recipes.

Does not own: `step`, `nifsVerify`, a deployment-application selection, Rust,
generated rows, a caller-provided validity proposition, or a new IR.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

/-- The paper-mandated application certificate for the Phase 3/4 calls.

The semantic propositions are not fields.  Each member is an executable
physical row program carrying the existing artifact-independent soundness,
honest-completeness, inactive-satisfiability, ownership, and receipt contract.
-/
structure ApplicationCertification (parameters : Parameters) where
  profile : Profile parameters
  hashPrior :
    CallRecipe (signature parameters) (profile.family parameters)
      Call.hashPrior
  hashNext :
    CallRecipe (signature parameters) (profile.family parameters)
      Call.hashNext
  runningCheck :
    CallRecipe (signature parameters) (profile.family parameters)
      Call.runningCheck
  freshCheck :
    CallRecipe (signature parameters) (profile.family parameters)
      Call.freshCheck

namespace ApplicationCertification

private theorem cost_extensionality
    {left right : Cost}
    (recurringRows :
      left.recurringRows = right.recurringRows)
    (committedColumns :
      left.committedColumns = right.committedColumns)
    (publicColumns :
      left.publicColumns = right.publicColumns)
    (auxiliaryColumns :
      left.auxiliaryColumns = right.auxiliaryColumns) :
    left = right := by
  cases left
  cases right
  simp_all

private theorem auxiliaryLayout_cost_exact (width : Nat) :
    (auxiliaryLayout width).cost = ⟨0, 0, 0, width⟩ := by
  induction width with
  | zero =>
      rfl
  | succ width inductionHypothesis =>
      change
        Cost.oneColumn Ownership.auxiliaryColumn +
            (auxiliaryLayout width).cost =
          ⟨0, 0, 0, width + 1⟩
      rw [inductionHypothesis]
      apply cost_extensionality <;>
        simp [Cost.oneColumn] <;>
        omega

private theorem publicLayout_cost_exact (width : Nat) :
    (publicLayout width).cost = ⟨0, 0, width, 0⟩ := by
  induction width with
  | zero =>
      rfl
  | succ width inductionHypothesis =>
      change
        Cost.oneColumn Ownership.publicColumn +
            (publicLayout width).cost =
          ⟨0, 0, width + 1, 0⟩
      rw [inductionHypothesis]
      apply cost_extensionality <;>
        simp [Cost.oneColumn] <;>
        omega

/-- The complete profile-indexed Phase 3/4 certificate.

All four executable recipes are constructed from one application profile.
No call outcome, relation-validity proposition, or deployment artifact is
accepted from the caller. -/
def poseidon23
    (parameters : Parameters)
    (profile : Poseidon23ApplicationProfile parameters) :
    ApplicationCertification parameters where
  profile := profile.toTerminalEqualityProfile.toProfile
  hashPrior := Poseidon23HashPriorRecipe.recipe parameters profile
  hashNext := Poseidon23HashNextRecipe.recipe parameters profile
  runningCheck :=
    RunningCheckRecipe.recipe parameters profile.toTerminalEqualityProfile
  freshCheck :=
    FreshCheckRecipe.recipe parameters profile.toTerminalEqualityProfile

@[simp] theorem poseidon23_hashPrior
    (parameters : Parameters)
    (profile : Poseidon23ApplicationProfile parameters) :
    (poseidon23 parameters profile).hashPrior =
      Poseidon23HashPriorRecipe.recipe parameters profile :=
  rfl

@[simp] theorem poseidon23_hashNext
    (parameters : Parameters)
    (profile : Poseidon23ApplicationProfile parameters) :
    (poseidon23 parameters profile).hashNext =
      Poseidon23HashNextRecipe.recipe parameters profile :=
  rfl

@[simp] theorem poseidon23_runningCheck
    (parameters : Parameters)
    (profile : Poseidon23ApplicationProfile parameters) :
    (poseidon23 parameters profile).runningCheck =
      RunningCheckRecipe.recipe parameters
        profile.toTerminalEqualityProfile :=
  rfl

@[simp] theorem poseidon23_freshCheck
    (parameters : Parameters)
    (profile : Poseidon23ApplicationProfile parameters) :
    (poseidon23 parameters profile).freshCheck =
      FreshCheckRecipe.recipe parameters
        profile.toTerminalEqualityProfile :=
  rfl

/-- Exact application-dependent call order.  The two hash occurrences remain
distinct because their result ownership differs. -/
def calls : List Call :=
  [.hashPrior, .hashNext, .runningCheck, .freshCheck]

theorem calls_exact :
    calls = [.hashPrior, .hashNext, .runningCheck, .freshCheck] :=
  rfl

/-- The profile-indexed Phase 3/4 slice invokes each certified call exactly
once.  These multiplicities are derived from the typed call list rather than
accepted as cost metadata. -/
theorem call_multiplicities :
    calls.count Call.hashPrior = 1 ∧
      calls.count Call.hashNext = 1 ∧
      calls.count Call.runningCheck = 1 ∧
      calls.count Call.freshCheck = 1 := by
  decide

/-- The prior hash allocates an auxiliary digest. -/
theorem hashPrior_output_ownership (parameters : Parameters) :
    (callOutputs parameters Call.hashPrior).map
        (fun port => port.layout.owners) =
      [List.replicate parameters.widths.digest
        Ownership.auxiliaryColumn] := by
  rfl

/-- The next hash allocates the public digest. -/
theorem hashNext_output_ownership (parameters : Parameters) :
    (callOutputs parameters Call.hashNext).map
        (fun port => port.layout.owners) =
      [List.replicate parameters.widths.digest
        Ownership.publicColumn] := by
  rfl

/-- The two hash calls cannot be interchanged: the prior digest is auxiliary
while the next digest is the public Step result. -/
theorem hash_outputs_distinct
    (parameters : Parameters)
    (profile : Poseidon23ApplicationProfile parameters) :
    (callOutputs parameters Call.hashPrior).map
          (fun port => port.layout.owners) ≠
      (callOutputs parameters Call.hashNext).map
          (fun port => port.layout.owners) := by
  have digestWidth : parameters.widths.digest = 5 :=
    (profile.toTerminalEqualityProfile.toProfile.digest_width_eq_codec
      parameters).trans profile.digestWidth
  rw [hashPrior_output_ownership, hashNext_output_ownership,
    digestWidth]
  decide

/-- Both terminal relations have their own typed call occurrence.  No NIFS
call is accepted as a substitute for either one. -/
theorem terminal_calls_distinct :
    Call.runningCheck ≠ Call.freshCheck := by
  decide

/-- Intrinsic resource use is computed from the certified signature rather
than from a Rust measurement.  Output ownership is included by `callCost`. -/
def hashPriorCost
    (parameters : Parameters)
    (_certificate : ApplicationCertification parameters) : Cost :=
  (signature parameters).callCost Call.hashPrior

def hashNextCost
    (parameters : Parameters)
    (_certificate : ApplicationCertification parameters) : Cost :=
  (signature parameters).callCost Call.hashNext

def runningCheckCost
    (parameters : Parameters)
    (_certificate : ApplicationCertification parameters) : Cost :=
  (signature parameters).callCost Call.runningCheck

def freshCheckCost
    (parameters : Parameters)
    (_certificate : ApplicationCertification parameters) : Cost :=
  (signature parameters).callCost Call.freshCheck

/-- Exact four-call Phase 3/4 cost, before typed-program multiplicities. -/
def phase34Cost
    (parameters : Parameters)
    (certificate : ApplicationCertification parameters) : Cost :=
  hashPriorCost parameters certificate +
    hashNextCost parameters certificate +
    runningCheckCost parameters certificate +
    freshCheckCost parameters certificate

theorem hashPriorCost_exact
    (parameters : Parameters)
    (profile : Poseidon23ApplicationProfile parameters) :
    hashPriorCost parameters (poseidon23 parameters profile) =
      ⟨2 * profile.alignmentWidth + profile.alignmentWidth.pred + 2503,
        0, 0,
        2 * profile.alignmentWidth + profile.alignmentWidth.pred + 2499⟩ := by
  have digestWidth : parameters.widths.digest = 5 :=
    (profile.toTerminalEqualityProfile.toProfile.digest_width_eq_codec
      parameters).trans profile.digestWidth
  unfold hashPriorCost Signature.callCost CallFootprint.cost Schema.cost
  rw [show (signature parameters).callFootprint Call.hashPrior =
      Poseidon23Hash.footprint profile.alignmentWidth by
    simpa [signature, callFootprint] using profile.hashFootprint]
  simp only [Poseidon23Hash.footprint, List.map_cons, List.map_nil,
    Cost.sum, auxiliaryLayout_cost_exact, signature, callOutputs,
    Ports.auxiliaryDigest, dataPort, Port.cost, digestWidth]
  apply cost_extensionality <;>
    simp only [Cost.add_recurringRows, Cost.add_committedColumns,
      Cost.add_publicColumns, Cost.add_auxiliaryColumns, Cost.zero] <;>
    omega

theorem hashNextCost_exact
    (parameters : Parameters)
    (profile : Poseidon23ApplicationProfile parameters) :
    hashNextCost parameters (poseidon23 parameters profile) =
      ⟨2 * profile.alignmentWidth + profile.alignmentWidth.pred + 2503,
        0, 5,
        2 * profile.alignmentWidth + profile.alignmentWidth.pred + 2494⟩ := by
  have digestWidth : parameters.widths.digest = 5 :=
    (profile.toTerminalEqualityProfile.toProfile.digest_width_eq_codec
      parameters).trans profile.digestWidth
  unfold hashNextCost Signature.callCost CallFootprint.cost Schema.cost
  rw [show (signature parameters).callFootprint Call.hashNext =
      Poseidon23Hash.footprint profile.alignmentWidth by
    simpa [signature, callFootprint] using profile.hashFootprint]
  simp only [Poseidon23Hash.footprint, List.map_cons, List.map_nil,
    Cost.sum, auxiliaryLayout_cost_exact, signature, callOutputs,
    Ports.publicDigest, dataPort, Port.cost, digestWidth]
  rw [publicLayout_cost_exact]
  apply cost_extensionality <;>
    simp only [Cost.add_recurringRows, Cost.add_committedColumns,
      Cost.add_publicColumns, Cost.add_auxiliaryColumns, Cost.zero] <;>
    omega

theorem runningCheckCost_exact
    (parameters : Parameters)
    (profile : Poseidon23ApplicationProfile parameters) :
    runningCheckCost parameters (poseidon23 parameters profile) =
      ⟨2 * profile.codecs.running.width +
          profile.codecs.running.width.pred + 1,
        0, 0,
        2 * profile.codecs.running.width +
          profile.codecs.running.width.pred + 1⟩ := by
  have bitWidth :
      parameters.widths.bit = 1 :=
    profile.toTerminalEqualityProfile.toProfile.bit_width_eq_one parameters
  unfold runningCheckCost Signature.callCost CallFootprint.cost Schema.cost
  rw [show (signature parameters).callFootprint Call.runningCheck =
      DirectCalls.equalityFootprint profile.codecs.running.width by
    simpa [signature, callFootprint] using profile.runningFootprint]
  simp only [DirectCalls.equalityFootprint, List.map_cons, List.map_nil,
    Cost.sum, auxiliaryLayout_cost_exact, signature, callOutputs,
    Ports.auxiliaryBit, bitPort, Port.cost, bitWidth]
  apply cost_extensionality <;>
    simp only [Cost.add_recurringRows, Cost.add_committedColumns,
      Cost.add_publicColumns, Cost.add_auxiliaryColumns, Cost.zero] <;>
    omega

theorem freshCheckCost_exact
    (parameters : Parameters)
    (profile : Poseidon23ApplicationProfile parameters) :
    freshCheckCost parameters (poseidon23 parameters profile) =
      ⟨2 * profile.codecs.fresh.width +
          profile.codecs.fresh.width.pred + 1,
        0, 0,
        2 * profile.codecs.fresh.width +
          profile.codecs.fresh.width.pred + 1⟩ := by
  have bitWidth :
      parameters.widths.bit = 1 :=
    profile.toTerminalEqualityProfile.toProfile.bit_width_eq_one parameters
  unfold freshCheckCost Signature.callCost CallFootprint.cost Schema.cost
  rw [show (signature parameters).callFootprint Call.freshCheck =
      DirectCalls.equalityFootprint profile.codecs.fresh.width by
    simpa [signature, callFootprint] using profile.freshFootprint]
  simp only [DirectCalls.equalityFootprint, List.map_cons, List.map_nil,
    Cost.sum, auxiliaryLayout_cost_exact, signature, callOutputs,
    Ports.auxiliaryBit, bitPort, Port.cost, bitWidth]
  apply cost_extensionality <;>
    simp only [Cost.add_recurringRows, Cost.add_committedColumns,
      Cost.add_publicColumns, Cost.add_auxiliaryColumns, Cost.zero] <;>
    omega

theorem phase34Cost_exact
    (parameters : Parameters)
    (profile : Poseidon23ApplicationProfile parameters) :
    phase34Cost parameters (poseidon23 parameters profile) =
      ⟨(2 * profile.alignmentWidth + profile.alignmentWidth.pred + 2503) +
          (2 * profile.alignmentWidth + profile.alignmentWidth.pred + 2503) +
          (2 * profile.codecs.running.width +
            profile.codecs.running.width.pred + 1) +
          (2 * profile.codecs.fresh.width +
            profile.codecs.fresh.width.pred + 1),
        0,
        5,
        (2 * profile.alignmentWidth + profile.alignmentWidth.pred + 2499) +
          (2 * profile.alignmentWidth + profile.alignmentWidth.pred + 2494) +
          (2 * profile.codecs.running.width +
            profile.codecs.running.width.pred + 1) +
          (2 * profile.codecs.fresh.width +
            profile.codecs.fresh.width.pred + 1)⟩ := by
  rw [phase34Cost, hashPriorCost_exact, hashNextCost_exact,
    runningCheckCost_exact, freshCheckCost_exact]
  rfl

end ApplicationCertification

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

import DirectCcsFPrime.FPrimeTraceAuthority

/-!
Weakness red-team checks for trace-carrying F' authority.

These checks isolate ways the trace-authority pattern can be misused. The
compiled theorems prove that the weak shapes are insufficient. The quarantined
comment block at the bottom contains examples that are meant to fail when
uncommented; they are kept here as regression probes for future strengthening.
-/

namespace DirectCcsFPrime

namespace FPrimeTraceAuthorityWeaknessRedTeam

/-- Tiny public-image universe for verifier-functionality attacks. -/
inductive BitImage where
  | zero
  | one
deriving DecidableEq

/-- A deliberately over-broad transition relation. -/
def AnyTransition : Nat → BitImage → BitImage → Prop :=
  fun _ _ _ => True

/--
One opaque proof is accepted for either one-step public image.

This models a verifier surface that proves reachability but does not bind a
single opaque proof to a unique `(steps, image)` pair.
-/
def VerifyStepOneAnyImage : Nat → Unit → BitImage → Prop :=
  fun steps _ _ => steps = 1

private def oneStepTrace (image : BitImage) :
    FPrimeTraceAuthority.Authority
      AnyTransition
      BitImage.zero
      1
      image where
  imageAt := fun i => if i = 0 then BitImage.zero else image
  startsAtInitial := by simp
  endsAtClaimed := by simp
  stepValid := by
    intro _i _hi
    trivial

private theorem verifyStepOneAnyImage_traceSound :
    FPrimeTraceAuthority.VerifierSound
      AnyTransition
      BitImage.zero
      VerifyStepOneAnyImage := by
  intro steps proof image hVerify
  subst steps
  exact ⟨oneStepTrace image⟩

/--
Trace soundness alone does not imply same-proof functionality.

The same opaque proof accepts both one-step images under the weak verifier.
-/
theorem trace_soundness_does_not_imply_same_proof_functionality :
    FPrimeTraceAuthority.VerifierSound
        AnyTransition
        BitImage.zero
        VerifyStepOneAnyImage ∧
      ¬ CompressedFPrimeAuthority.ProofFunctional VerifyStepOneAnyImage := by
  constructor
  · exact verifyStepOneAnyImage_traceSound
  · intro hFunctional
    rcases
      hFunctional
        (stepsA := 1)
        (stepsB := 1)
        (proof := ())
        (imageA := BitImage.zero)
        (imageB := BitImage.one)
        rfl
        rfl with
      ⟨_hSteps, hImages⟩
    cases hImages

/--
The weak verifier is trace-sound for the universal transition.

This is intentional: with `AnyTransition`, every one-step image has a trace.
The red-team failure is missing same-proof functionality, not missing
trace soundness.
-/
theorem verifier_sound_for_universal_transition :
    FPrimeTraceAuthority.VerifierSound
      AnyTransition
      BitImage.zero
      VerifyStepOneAnyImage :=
  verifyStepOneAnyImage_traceSound

/--
A universal transition relation makes trace authority useless as a protocol
check.

This does not break Lean soundness; it shows that the production `Transition`
must include the real F' step obligations, including the parent-only DEC/stage
facts.
-/
theorem universal_transition_authorizes_any_one_step_image :
    ∀ image : BitImage,
      Nonempty
        (FPrimeTraceAuthority.Authority
          AnyTransition
          BitImage.zero
          1
          image) := by
  intro image
  exact ⟨oneStepTrace image⟩

/-- Constant digest for a two-parent toy model. -/
def ConstantParentHash (_parent : Bool) : Unit :=
  ()

/-- Binding property for a toy parent-hash model. -/
def ToyParentHashBinding (hash : Bool → Unit) : Prop :=
  ∀ a b : Bool, hash a = hash b → a = b

/--
A constant parent hash cannot bind two possible parent handles.

This is the tiny concrete reminder that the Poseidon2 parent binding object is
a real cryptographic assumption, not a definitional fact.
-/
theorem constant_parent_hash_cannot_bind_parent_handles :
    ¬ ToyParentHashBinding ConstantParentHash := by
  intro hBinding
  have hFalseTrue : false = true :=
    hBinding false true rfl
  cases hFalseTrue

/-
Quarantined break probes.

Uncomment these examples when tightening the corresponding boundary. Each one
is intentionally false for the weak model above, so uncommenting it should
break the build until the model is replaced by a stronger production surface.

Do not add `¬ VerifierSound AnyTransition ...` here. That claim is false for
the toy model because the transition relation accepts every edge. The useful
negative probe is same-proof functionality.

example :
    CompressedFPrimeAuthority.ProofFunctional VerifyStepOneAnyImage := by
  intro hA hB

example :
    ToyParentHashBinding ConstantParentHash := by
  intro a b h
-/

end FPrimeTraceAuthorityWeaknessRedTeam

end DirectCcsFPrime

import Nightstream.Implementation.Rust.CanonicalConformance.OneSlot.Generated.StepCases
import Nightstream.Implementation.Rust.CanonicalConformance.OneSlot.Generated.TerminalCases

/-!
Contract: exact logical boundary of the generated one-slot Rust differential
corpora.

Owns:
- input-only normalization that erases the externally observed Rust Boolean;
- the proposition that an arbitrary acceptance oracle agrees with every
  generated Step or Terminal input;
- explicit oracles that agree on the complete generated corpora but disagree
  with the canonical checker on one fresh input.

Does not own: Rust semantics, a source-to-Lean translation, production
refinement, R1CS, or a claim that the generated cases are unhelpful.  The
corpora remain valid bounded differential evidence.  These countermodels prove
only that finite agreement cannot by itself establish the universal
obligation-11 acceptance theorem.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Rust.CanonicalConformance.OneSlot.FiniteCorpusBoundary

open Nightstream.Implementation.Rust.CanonicalConformance.OneSlot

namespace Step

/-- Remove the externally observed result so oracle inputs contain only the
shared verifier input and primitive receipts. -/
def inputOnly (case : StepCase) : StepCase :=
  { case with rustAccepted := false }

abbrev AcceptanceOracle := StepCase -> Bool

def canonicalOracle : AcceptanceOracle :=
  stepAccepted

/-- Agreement with every generated shared Step input, after erasing the
recorded Rust result from the oracle domain. -/
def ConformsOnGenerated (oracle : AcceptanceOracle) : Prop :=
  ∀ case,
    case ∈
        Nightstream.Implementation.Rust.CanonicalConformance.OneSlot.Generated.all ->
      oracle (inputOnly case) = canonicalOracle (inputOnly case)

/-- A concrete input outside the generated Step corpus.  Only the iteration is
fresh; the external Rust result has already been erased. -/
def outsider : StepCase :=
  { inputOnly
      Nightstream.Implementation.Rust.CanonicalConformance.OneSlot.Generated.honestBase with
    iteration := 1000 }

theorem outsider_not_mem_generated_inputs :
    outsider ∉
      Nightstream.Implementation.Rust.CanonicalConformance.OneSlot.Generated.all.map
        inputOnly := by
  decide

/-- An adversarial acceptance oracle that differs at exactly one unobserved
input. -/
def flippedOutside : AcceptanceOracle :=
  fun input =>
    if input = outsider then
      !canonicalOracle input
    else
      canonicalOracle input

theorem flippedOutside_conformsOnGenerated :
    ConformsOnGenerated flippedOutside := by
  intro case member
  have inputMember :
      inputOnly case ∈
        Nightstream.Implementation.Rust.CanonicalConformance.OneSlot.Generated.all.map
          inputOnly :=
    List.mem_map.mpr ⟨case, member, rfl⟩
  have inputNe : inputOnly case ≠ outsider := by
    intro equal
    apply outsider_not_mem_generated_inputs
    rw [← equal]
    exact inputMember
  simp [flippedOutside, inputNe]

theorem flippedOutside_disagrees :
    flippedOutside outsider ≠ canonicalOracle outsider := by
  simp [flippedOutside]

/-- The invalid inference that generated corpus agreement determines the
acceptance function on every Step input. -/
def AttemptedUniversalBridge : Prop :=
  ∀ oracle : AcceptanceOracle,
    ConformsOnGenerated oracle ->
      ∀ input, oracle input = canonicalOracle input

/-- Kernel-checked obstruction: the exact generated Step corpus cannot alone
prove universal production acceptance refinement. -/
theorem not_attemptedUniversalBridge :
    ¬ AttemptedUniversalBridge := by
  intro attempted
  exact flippedOutside_disagrees
    (attempted flippedOutside flippedOutside_conformsOnGenerated outsider)

end Step

namespace Terminal

/-- Remove the externally observed result so oracle inputs contain only the
shared terminal input and primitive receipts. -/
def inputOnly (case : TerminalCase) : TerminalCase :=
  { case with rustAccepted := false }

abbrev AcceptanceOracle := TerminalCase -> Bool

def canonicalOracle : AcceptanceOracle :=
  terminalAccepted

/-- Agreement with every generated shared Terminal input. -/
def ConformsOnGenerated (oracle : AcceptanceOracle) : Prop :=
  ∀ case,
    case ∈
        Nightstream.Implementation.Rust.CanonicalConformance.OneSlot.Generated.Terminal.all ->
      oracle (inputOnly case) = canonicalOracle (inputOnly case)

/-- A concrete input outside the generated Terminal corpus. -/
def outsider : TerminalCase :=
  { inputOnly
      Nightstream.Implementation.Rust.CanonicalConformance.OneSlot.Generated.Terminal.honestBase with
    iteration := 1000 }

theorem outsider_not_mem_generated_inputs :
    outsider ∉
      Nightstream.Implementation.Rust.CanonicalConformance.OneSlot.Generated.Terminal.all.map
        inputOnly := by
  decide

def flippedOutside : AcceptanceOracle :=
  fun input =>
    if input = outsider then
      !canonicalOracle input
    else
      canonicalOracle input

theorem flippedOutside_conformsOnGenerated :
    ConformsOnGenerated flippedOutside := by
  intro case member
  have inputMember :
      inputOnly case ∈
        Nightstream.Implementation.Rust.CanonicalConformance.OneSlot.Generated.Terminal.all.map
          inputOnly :=
    List.mem_map.mpr ⟨case, member, rfl⟩
  have inputNe : inputOnly case ≠ outsider := by
    intro equal
    apply outsider_not_mem_generated_inputs
    rw [← equal]
    exact inputMember
  simp [flippedOutside, inputNe]

theorem flippedOutside_disagrees :
    flippedOutside outsider ≠ canonicalOracle outsider := by
  simp [flippedOutside]

def AttemptedUniversalBridge : Prop :=
  ∀ oracle : AcceptanceOracle,
    ConformsOnGenerated oracle ->
      ∀ input, oracle input = canonicalOracle input

/-- Kernel-checked obstruction: the exact generated Terminal corpus cannot
alone prove universal production acceptance refinement. -/
theorem not_attemptedUniversalBridge :
    ¬ AttemptedUniversalBridge := by
  intro attempted
  exact flippedOutside_disagrees
    (attempted flippedOutside flippedOutside_conformsOnGenerated outsider)

end Terminal

end Nightstream.Implementation.Rust.CanonicalConformance.OneSlot.FiniteCorpusBoundary

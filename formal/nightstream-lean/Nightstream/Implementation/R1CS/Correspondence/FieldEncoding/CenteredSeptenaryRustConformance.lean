import Nightstream.Implementation.R1CS.Artifacts.CenteredSeptenary
import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.CenteredSeptenary

/-!
Exact finite-corpus Rust/Lean check for the centered-septenary encoder.

Assurance tier: Rust-conformant for the four generated boundary cases only.

Owns: exact comparison of the Rust-generated digit vectors with the Lean
encoder for zero, one, the Goldilocks midpoint, and the largest canonical
Goldilocks residue.

Does not own: universal Rust-function equivalence, generated F-prime
assignment conformance, the outer radix-four norm premise, or relation
soundness.

Emits constraints: no.
-/

namespace Nightstream.Implementation.R1CS.CenteredSeptenaryRustConformance

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.CenteredSeptenaryField

namespace Artifact

open Nightstream.Implementation.R1CS.Artifacts.CenteredSeptenary.RustEncoderCases

abbrev sources :=
  Nightstream.Implementation.R1CS.Artifacts.CenteredSeptenary.RustEncoderCases.sources

abbrev cases :=
  Nightstream.Implementation.R1CS.Artifacts.CenteredSeptenary.RustEncoderCases.cases

end Artifact

def boundarySources : List Nat :=
  [0, 1, goldilocksP / 2, goldilocksP - 1]

def encodeList (source : Nat) : List Nat :=
  (List.range digitCount).map (encodeDigit source)

theorem generated_sources_are_exact_boundaries :
    Artifact.sources = boundarySources := by
  decide

theorem generated_cases_match_lean_encoder :
    Artifact.cases =
      boundarySources.map (fun source => (source, encodeList source)) := by
  decide

theorem generated_case_count : Artifact.cases.length = 4 := by
  decide

theorem generated_digit_counts :
    Artifact.cases.all (fun case => case.2.length = digitCount) = true := by
  decide

end Nightstream.Implementation.R1CS.CenteredSeptenaryRustConformance


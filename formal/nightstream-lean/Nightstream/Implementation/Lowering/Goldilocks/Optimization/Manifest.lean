import Nightstream.Implementation.Lowering.Goldilocks.CanonicalManifest
import Nightstream.Implementation.Lowering.Goldilocks.Optimization.R1CS

/-!
Contract: certify the existing canonical proof-free manifest as an exact
identity optimization boundary.

Assurance tier: model-level.

Owns: the source-encoding to decoded-manifest replacement theorem, exact
receipt cost, and normalized-row acceptance.

Does not own: JSON, Rust decoding, a cost reduction, or a protocol-specific
optimization.

Emits constraints: the normalized rows already stored by the canonical
manifest.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.Goldilocks.Optimization.Manifest

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.CanonicalManifest

universe u

private abbrev Assignment := R1CS.Assignment

def decodedSystem
    {Observable : Type u}
    (program : CanonicalManifest.Program)
    (observe : Assignment -> Observable) :=
  R1CS.system program.decode.one program.decode.rows observe

/-- The canonical manifest is already the normalized identity pass. -/
def ofEncodingReplacement
    {Observable : Type u}
    {signature : Nightstream.Implementation.Lowering.Typed.Signature}
    {input output : Nightstream.Implementation.Lowering.Typed.Schema
      signature.types}
    {source :
      Nightstream.Implementation.Lowering.Typed.Program
        signature input output}
    (encoding : Encoding source)
    (observe : Assignment -> Observable)
    (degreeLimit : Nat)
    (withinLimit : R1CS.degree <= degreeLimit) :
    Optimization.Replacement
      (R1CS.ofEncoding encoding observe)
      (decodedSystem (CanonicalManifest.Program.ofEncoding encoding) observe)
      degreeLimit where
  recover := fun assignment => assignment
  derive := fun assignment => assignment
  sound := by
    intro assignment accepted
    exact ⟨accepted.1,
      (CanonicalManifest.Program.decoded_satisfies_iff
        encoding assignment).mp accepted.2⟩
  complete := by
    intro assignment accepted
    exact ⟨accepted.1,
      (CanonicalManifest.Program.decoded_satisfies_iff
        encoding assignment).mpr accepted.2⟩
  recover_observes := fun _ _ => rfl
  derive_observes := fun _ _ => rfl
  source_degree := withinLimit
  target_degree := withinLimit

theorem cost_exact
    {signature : Nightstream.Implementation.Lowering.Typed.Signature}
    {input output : Nightstream.Implementation.Lowering.Typed.Schema
      signature.types}
    {source :
      Nightstream.Implementation.Lowering.Typed.Program
        signature input output}
    (encoding : Encoding source) :
    (CanonicalManifest.Program.ofEncoding encoding).cost = encoding.cost :=
  CanonicalManifest.Program.cost_ofEncoding encoding

theorem rows_exact
    {signature : Nightstream.Implementation.Lowering.Typed.Signature}
    {input output : Nightstream.Implementation.Lowering.Typed.Schema
      signature.types}
    {source :
      Nightstream.Implementation.Lowering.Typed.Program
        signature input output}
    (encoding : Encoding source) :
    (CanonicalManifest.Program.ofEncoding encoding).rows.length =
      encoding.rows.length :=
  CanonicalManifest.Program.rows_length_ofEncoding encoding

end Nightstream.Implementation.Lowering.Goldilocks.Optimization.Manifest

/-
Generated file: production combined-NC artifact; do not hand-edit.

Owns: the exact source-program and source-decoder family roots.

Does not own: decoding, row satisfaction, transcript authority, commitment
binding, semantic acceptance, costs, or permission to remove rows.

Emits constraints: no.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.generated` | The generated payload named by `Owns` above | computed artifact |
-/

import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Provenance.SourceColumns
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Provenance.RetainedSlots
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Provenance.LinearDefinitions
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Provenance.TraceEliminatedColumns
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Provenance.DerivedProductSums
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Provenance.RewriteSteps
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Provenance.RetainedSteps
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Provenance.Decoders

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Provenance

def sourceArm : Nat := 2
def decoderArm : Nat := 2

def sourceColumns : List Nat := SourceColumns.values
def retainedSlots : List RawSourceSlot := RetainedSlots.values
def linearDefinitions : List RawSourceDefinition := LinearDefinitions.values
def traceEliminatedColumns : List Nat := TraceEliminatedColumns.values
def derivedProductSums : List RawDerivedProductSum := DerivedProductSums.values
def rewriteSteps : List RawRewriteStep := RewriteSteps.values
def retainedSteps : List RawRetainedStep := RetainedSteps.values
def decoders : List RawSourceDecoder := Decoders.values

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Provenance

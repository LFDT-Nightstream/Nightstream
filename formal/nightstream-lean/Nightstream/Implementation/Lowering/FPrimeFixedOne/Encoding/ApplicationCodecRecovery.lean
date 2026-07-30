import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Profile
import Nightstream.Implementation.Lowering.Goldilocks.CodecRecovery

/-!
Contract: certify exact-width recovery for the four codecs that HyperNova
leaves to the deployment application.

Owns: recovery of application state, step witness, running-relation witness,
and fresh-relation witness values from exact-width field coordinates.

Does not own: canonical running, fresh, or NIFS-proof codecs; physical rows;
application semantics; Rust; or generated artifacts.

Emits constraints: none.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.Goldilocks

/-- A complete physical deployment must interpret every exact-width value of
each application-owned input codec. Protocol-owned NIFS carriers are proved
recoverable in their own modules and are not caller fields here. -/
structure ApplicationCodecRecovery
    (parameters : Parameters)
    (codecs : DataCodecs parameters) : Prop where
  state : codecs.state.ExactWidthRecoverable
  witness : codecs.witness.ExactWidthRecoverable
  runningWitness : codecs.runningWitness.ExactWidthRecoverable
  freshWitness : codecs.freshWitness.ExactWidthRecoverable

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

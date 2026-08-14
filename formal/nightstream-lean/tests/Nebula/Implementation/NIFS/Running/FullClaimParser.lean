import Nightstream.Implementation.Nebula.NIFS.Running.FullClaimParser

set_option autoImplicit false
set_option maxRecDepth 100000

namespace tests.NebulaProductFullClaimParser

open Nightstream.Implementation.Nebula.FullClaimEnvelope
open Nightstream.Implementation.Nebula.ProductFullClaimParser
open Nightstream.Implementation.Nebula.ProductNifsCodec
open Nightstream.Protocol.Nebula.WasmPublicStatementEncoding
open Nightstream.SuperNeo.Concrete.Phi81Relation

example {fullShape : Shape}
    (contract : FullShapeContract fullShape)
    (application : PublicImage)
    (value : Value
      Nightstream.Implementation.Nebula.ProductFullClaimDecoder.widths)
    (wellFormed : WellFormed contract application value) :
    parseValue contract application value.block =
      some (value, runningOfValue contract value) :=
  parseValue_block contract application value wellFormed

example {fullShape : Shape}
    (contract : FullShapeContract fullShape)
    (application : PublicImage)
    {block : Block}
    {value : Value
      Nightstream.Implementation.Nebula.ProductFullClaimDecoder.widths}
    {running : Running fullShape}
    (accepted : parseValue contract application block =
      some (value, running)) :
    WellFormed contract application value ∧
      value.block = block :=
  ⟨(parseValue_success contract application accepted).1,
    (parseValue_success contract application accepted).2.1⟩

#check decode_success
#check parseValue_rejects_profile_mismatch
#check parseValue_rejects_application_mismatch
#check parseValue_rejects_bundle_failure
#check parseValue_rejects_running_failure
#check parseValue_rejects_memory_failure
#check claimDecoder

end tests.NebulaProductFullClaimParser

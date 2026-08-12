import Nightstream.Assurance.NebulaV2.ReleasePipeline

set_option autoImplicit false

namespace Nightstream.Tests.NebulaV2ReleasePipeline

open Nightstream.Assurance.NebulaV2.ReleasePipeline

theorem accepted_bytes_have_one_terminal_parse
    {Bytes Parsed : Type}
    {decode : Bytes → Option Parsed}
    {terminalAccepts : Parsed → Prop}
    {proof : Bytes}
    (accepted : Accepts decode terminalAccepts proof) :
    ∃ parsed, decode proof = some parsed ∧ terminalAccepts parsed :=
  accepted

end Nightstream.Tests.NebulaV2ReleasePipeline

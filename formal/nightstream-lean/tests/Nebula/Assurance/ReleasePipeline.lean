import Nightstream.Assurance.Nebula.ReleasePipeline

set_option autoImplicit false

namespace Nightstream.Tests.NebulaReleasePipeline

open Nightstream.Assurance.Nebula.ReleasePipeline

theorem accepted_bytes_have_one_terminal_parse
    {Bytes Parsed : Type}
    {decode : Bytes → Option Parsed}
    {terminalAccepts : Parsed → Prop}
    {proof : Bytes}
    (accepted : Accepts decode terminalAccepts proof) :
    ∃ parsed, decode proof = some parsed ∧ terminalAccepts parsed :=
  accepted

end Nightstream.Tests.NebulaReleasePipeline

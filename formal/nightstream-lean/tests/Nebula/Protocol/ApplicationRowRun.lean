import Nightstream.Protocol.Nebula.ApplicationRowRun

set_option autoImplicit false

namespace tests.NebulaApplicationRowRun

open Nightstream.Protocol.Nebula.ApplicationRowRun
open Nightstream.Protocol.Nebula.ApplicationTrace
open Nightstream.Protocol.Nebula.Completion
open Nightstream.Protocol.Nebula.Ports

variable {Program ApplicationState : Type}
variable {semantics : Semantics Program ApplicationState}
variable {program : Program}

/-- A run cannot manufacture termination without checking one terminal row. -/
theorem empty_rows_cannot_terminate
    {initial final : ApplicationState} {outcome : Outcome}
    (run : Runs semantics program (.running initial) []
      (.terminal final outcome)) : False := by
  cases run

/-- Once terminal, the local relation cannot consume an active row. -/
theorem active_after_terminal_is_rejected
    {state : ApplicationState} {outcome : Outcome}
    {row : NormalizedRow} {after : Phase ApplicationState}
    (run : Runs semantics program (.terminal state outcome)
      [.active row] after) : False := by
  cases run with
  | cons head tail => cases head

/-- The soundness bridge exposes the exact active prefix and typed terminal
transition; neither object is an input to the bridge theorem. -/
theorem local_rows_reconstruct_semantic_shape
    {initial final : ApplicationState} {outcome : Outcome}
    {rows : List ApplicationRow}
    (run : Runs semantics program (.running initial) rows
      (.terminal final outcome)) :
    exists (activeRows : List NormalizedRow)
        (beforeTerminal : ApplicationState) (terminalRow : NormalizedRow)
        (paddingCount : Nat),
      rows = activeRows.map ApplicationRow.active ++
        [terminalApplicationRow terminalRow outcome] ++
        List.replicate paddingCount .padding /\
      ActivePrefix semantics program initial activeRows beforeTerminal /\
      Terminal semantics program beforeTerminal final terminalRow outcome :=
  run.complete_inverse

end tests.NebulaApplicationRowRun

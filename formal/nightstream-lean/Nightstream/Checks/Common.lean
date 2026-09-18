namespace Nightstream.Checks

/-- A named executable assertion. The thunk keeps expensive checks lazy until
the runner is ready to print their result. -/
structure Probe where
  name : String
  evaluate : Unit → Bool
  expected : Bool

private def flush : IO Unit :=
  IO.getStdout >>= IO.FS.Stream.flush

/-- Evaluate and print one probe before moving to the next one. -/
def runProbe (probe : Probe) : IO Bool := do
  let pass := probe.evaluate () == probe.expected
  IO.println s!"{probe.name}={pass}"
  flush
  pure pass

/-- Run probes sequentially without first evaluating the rest of the group. -/
def runProbes (probes : List Probe) : IO Bool := do
  let mut ok := true
  for probe in probes do
    unless ← runProbe probe do
      ok := false
  pure ok

end Nightstream.Checks

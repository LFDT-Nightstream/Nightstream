import Nightstream.Checks.Envelope
import Nightstream.Checks.Protocol
import Nightstream.Checks.Rust

/-!
`lake exe check` is the executable assurance gate.

Probe groups and individual expensive checks run sequentially, and each result
is flushed before the next check starts. Symbol anchors are conformance evidence,
not proof authority. Any failed assertion or missing anchor exits nonzero.
-/

private def flush : IO Unit :=
  IO.getStdout >>= IO.FS.Stream.flush

def main : IO UInt32 := do
  let envelopeOk ← Nightstream.Checks.Envelope.run
  let protocolOk ← Nightstream.Checks.Protocol.run
  let rustOk ← Nightstream.Checks.Rust.run
  let ok := envelopeOk && protocolOk && rustOk
  if ok then
    IO.println "check=pass"
    flush
    return 0
  else
    IO.println "check=FAIL"
    flush
    return 1

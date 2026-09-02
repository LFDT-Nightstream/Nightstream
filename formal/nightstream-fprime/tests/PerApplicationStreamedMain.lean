import NightstreamFPrime.Export.Main
import tests.PerApplicationEmitterFixture

/-! Native test entrypoint for the generic streamed package writer. -/

namespace NightstreamFPrime.Tests.PerApplicationStreamedMain

open NightstreamFPrime.Export

def run (arguments : List String) : IO UInt32 := do
  match arguments with
  | [path] | ["--", path] =>
      let start ← IO.monoMsNow
      Main.emitPerApplication (PerApplicationEmitterFixture.program ())
        (PerApplicationEmitterFixture.fits ()) ⟨path⟩
      let finish ← IO.monoMsNow
      IO.println s!"per_application_streamed_ms={finish - start}"
      pure 0
  | _ =>
      IO.eprintln "usage: emitPerApplicationStreamedFixture <path>"
      pure 2

end NightstreamFPrime.Tests.PerApplicationStreamedMain

def main (arguments : List String) : IO UInt32 :=
  NightstreamFPrime.Tests.PerApplicationStreamedMain.run arguments

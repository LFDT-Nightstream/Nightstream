import NightstreamFPrime.Export.Main

/-! Executable wrapper for the canonical Stage 1 package emitter. -/

def main (arguments : List String) : IO UInt32 :=
  NightstreamFPrime.Export.Main.run arguments

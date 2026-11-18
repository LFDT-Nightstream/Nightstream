import NightstreamFPrime.Export.MemoryTraceParity
import NightstreamFPrime.Export.ParityEmitter

def main (arguments : List String) : IO UInt32 :=
  NightstreamFPrime.Export.ParityEmitter.run "emitted_memory_trace_parity"
    NightstreamFPrime.Export.MemoryTraceParity.parityValue arguments

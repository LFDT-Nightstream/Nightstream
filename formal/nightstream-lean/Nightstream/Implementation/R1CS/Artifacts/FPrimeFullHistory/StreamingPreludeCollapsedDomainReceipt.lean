import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPreludeCollapsedDomainReceipt

/-!
Facade for the Rust-emitted bounded Prelude domain-collapse receipt.

Owns the handwritten import boundary only. The receipt is non-authoritative
until its correspondence leaf checks every reference round.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeCollapsedDomainReceipt

def inputValues : List Nat :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPreludeCollapsedDomainReceipt.inputValues

def collapsedValues : List Nat :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPreludeCollapsedDomainReceipt.collapsedValues

def initialStates : List (List Nat) :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPreludeCollapsedDomainReceipt.initialStates

def partialStates : List (List Nat) :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPreludeCollapsedDomainReceipt.partialStates

def terminalStates : List (List Nat) :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPreludeCollapsedDomainReceipt.terminalStates

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeCollapsedDomainReceipt

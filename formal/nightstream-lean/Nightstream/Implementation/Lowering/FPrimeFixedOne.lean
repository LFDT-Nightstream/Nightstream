import Nightstream.Implementation.Lowering.Typed
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Step
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal

/-!
Frozen facade for obligation 9 of the paper-authoritative fixed-one `F'`
verifier.

This surface exports the artifact-independent typed IR, its total ownership
receipts and definitional cost accounting, and the exact step and terminal
programs.  It does not select a physical field encoding, claim Rust or R1CS
conformance, or assign authority to generated artifacts.  Those are separate
obligations 10 and 11.
-/

import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.ActiveSourceLayout.YZcolConsumer

/-!
Focused theorem-shape checks for the typed PiCCS → PiRLC `y_zcol` dataflow
bridge.
-/

open Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout

#check YZcolConsumer.ConsumerColumns
#check YZcolConsumer.ConsumerMatches
#check YZcolConsumer.decodedInputs
#check YZcolConsumer.decodedInputs_eq_yZcol_of_bound

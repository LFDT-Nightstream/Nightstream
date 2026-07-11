import Nightstream.Implementation.R1CS.EqualityPins

/-! Generated exact adjacent-state equality rows for the two-step full-history profile. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStateLink

open Nightstream.Implementation.R1CS

def pairs : List (Nat × Nat) := [(2, 10834), (3, 10835), (4, 10836), (5, 10837), (6, 10838), (7, 10839), (8, 10840), (9, 10841), (5947, 10842), (5948, 10843), (12, 10844), (13, 10845), (14, 10846), (15, 10847), (32, 10848), (33, 10849), (34, 10850), (35, 10851), (1, 10833), (5943, 10852), (5944, 10853), (5945, 10854), (5946, 10855), (5939, 10856), (5940, 10857), (5941, 10858), (5942, 10859), (32, 10860), (33, 10861), (34, 10862), (35, 10863)]
def rows : List Row := EqualityPins.rows pairs
def rowStart : Nat := 929981
def rowEnd : Nat := 930012
def rowCount : Nat := 31

theorem rows_length : rows.length = rowCount := by decide

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStateLink

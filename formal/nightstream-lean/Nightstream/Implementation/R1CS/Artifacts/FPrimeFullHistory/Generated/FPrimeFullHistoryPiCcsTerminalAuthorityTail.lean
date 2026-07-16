import Nightstream.Implementation.R1CS.Core.AffinePins

/-! Generated exact affine-pin phase. Do not hand-edit. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryPiCcsTerminalAuthorityTail

open Nightstream.Implementation.R1CS

def rangeSha256 : String := "b0ac1fadb2a570621fa932a25e45d4aa77cca22bb2a754333b7901afae725c14"
def rowStart : Nat := 1134885
def rowEnd : Nat := 1136175
def rowCount : Nat := 1290

def pinRuns : List AffinePins.Run :=
  [ .equal 1160249 1159865 1 1 2
  , .equal 1160251 1159993 1 1 2
  , .equal 1160253 1160121 1 1 2
  , .zero 1159973 1 20
  , .zero 1160101 1 20
  , .zero 1160229 1 20
  , .zero 1160363 1 20
  , .equal 1162039 1161655 1 1 2
  , .equal 1162041 1161783 1 1 2
  , .equal 1162043 1161911 1 1 2
  , .zero 1161763 1 20
  , .zero 1161891 1 20
  , .zero 1162019 1 20
  , .zero 1162153 1 20
  , .equal 1163829 1163445 1 1 2
  , .equal 1163831 1163573 1 1 2
  , .equal 1163833 1163701 1 1 2
  , .zero 1163553 1 20
  , .zero 1163681 1 20
  , .zero 1163809 1 20
  , .zero 1163943 1 20
  , .equal 1165619 1165235 1 1 2
  , .equal 1165621 1165363 1 1 2
  , .equal 1165623 1165491 1 1 2
  , .zero 1165343 1 20
  , .zero 1165471 1 20
  , .zero 1165599 1 20
  , .zero 1165733 1 20
  , .equal 1167409 1167025 1 1 2
  , .equal 1167411 1167153 1 1 2
  , .equal 1167413 1167281 1 1 2
  , .zero 1167133 1 20
  , .zero 1167261 1 20
  , .zero 1167389 1 20
  , .zero 1167523 1 20
  , .equal 1169199 1168815 1 1 2
  , .equal 1169201 1168943 1 1 2
  , .equal 1169203 1169071 1 1 2
  , .zero 1168923 1 20
  , .zero 1169051 1 20
  , .zero 1169179 1 20
  , .zero 1169313 1 20
  , .equal 1170989 1170605 1 1 2
  , .equal 1170991 1170733 1 1 2
  , .equal 1170993 1170861 1 1 2
  , .zero 1170713 1 20
  , .zero 1170841 1 20
  , .zero 1170969 1 20
  , .zero 1171103 1 20
  , .equal 1172779 1172395 1 1 2
  , .equal 1172781 1172523 1 1 2
  , .equal 1172783 1172651 1 1 2
  , .zero 1172503 1 20
  , .zero 1172631 1 20
  , .zero 1172759 1 20
  , .zero 1172893 1 20
  , .equal 1174569 1174185 1 1 2
  , .equal 1174571 1174313 1 1 2
  , .equal 1174573 1174441 1 1 2
  , .zero 1174293 1 20
  , .zero 1174421 1 20
  , .zero 1174549 1 20
  , .zero 1174683 1 20
  , .equal 1176359 1175975 1 1 2
  , .equal 1176361 1176103 1 1 2
  , .equal 1176363 1176231 1 1 2
  , .zero 1176083 1 20
  , .zero 1176211 1 20
  , .zero 1176339 1 20
  , .zero 1176473 1 20
  , .equal 1178149 1177765 1 1 2
  , .equal 1178151 1177893 1 1 2
  , .equal 1178153 1178021 1 1 2
  , .zero 1177873 1 20
  , .zero 1178001 1 20
  , .zero 1178129 1 20
  , .zero 1178263 1 20
  , .equal 1179939 1179555 1 1 2
  , .equal 1179941 1179683 1 1 2
  , .equal 1179943 1179811 1 1 2
  , .zero 1179663 1 20
  , .zero 1179791 1 20
  , .zero 1179919 1 20
  , .zero 1180053 1 20
  , .equal 1181729 1181345 1 1 2
  , .equal 1181731 1181473 1 1 2
  , .equal 1181733 1181601 1 1 2
  , .zero 1181453 1 20
  , .zero 1181581 1 20
  , .zero 1181709 1 20
  , .zero 1181843 1 20
  , .equal 1183519 1183135 1 1 2
  , .equal 1183521 1183263 1 1 2
  , .equal 1183523 1183391 1 1 2
  , .zero 1183243 1 20
  , .zero 1183371 1 20
  , .zero 1183499 1 20
  , .zero 1183633 1 20
  , .equal 1185309 1184925 1 1 2
  , .equal 1185311 1185053 1 1 2
  , .equal 1185313 1185181 1 1 2
  , .zero 1185033 1 20
  , .zero 1185161 1 20
  , .zero 1185289 1 20
  , .zero 1185423 1 20
  ]

def pins : List AffinePins.Pin := AffinePins.expandRuns pinRuns
def rows : List Row := AffinePins.rows pins

theorem pins_canonical : AffinePins.PinsCanonical pins := by native_decide
theorem pins_length : pins.length = rowCount := by
rw [pins, AffinePins.expandRuns_length]
native_decide

theorem rows_length : rows.length = rowCount := by
simpa [rows, AffinePins.rows] using pins_length

end Nightstream.Implementation.R1CS.FPrimeFullHistoryPiCcsTerminalAuthorityTail

import Nightstream.Implementation.R1CS.PiDecStrictCompiler

/-! Generated exact strict-PiDEC semantic wire layout. Do not hand-edit. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryPiDec

def layout : PiDecStrictCompiler.Layout := {
radix := 2
ringDimension := 54
extensionLimbs := 2
firstAllocatedColumn := 24781
parent := {
      commitment := { dCol := 24391, kappaCol := 24393, dataCols := ((List.range 972).map (fun index => 1 + 15 * index)) }
      adv := none
      xActiveCols := ((List.range 270).map (fun index => 14581 + 15 * index))
      xInactiveCol := 24766
      xRows := 54
      xWidth := 257
      xRowsCol := 24395
      xWidthCol := 24397
      mIn := 257
      mInCol := 24399
      yRingCols :=
        [((List.range 128).map (fun index => 18631 + 15 * index)),
        ((List.range 128).map (fun index => 20551 + 15 * index)),
        ((List.range 128).map (fun index => 22471 + 15 * index))]
      ctCols := [(28561, 28562), (28563, 28564), (28565, 28566)]
      rCols := [(24466, 24468)]
      sColCols := [(24496, 24498), (24500, 24502), (24504, 24506), (24508, 24510), (24512, 24514), (24516, 24518), (24520, 24522), (24524, 24526), (24528, 24530)]
      foldDigestCols := ((List.range 4).map (fun index => 28652 + 2 * index)) }
children :=
[    {
      commitment := { dCol := 24392, kappaCol := 24394, dataCols := ((List.range 972).map (fun index => 2 + 15 * index)) }
      adv := none
      xActiveCols := ((List.range 270).map (fun index => 14582 + 15 * index))
      xInactiveCol := 24767
      xRows := 54
      xWidth := 257
      xRowsCol := 24396
      xWidthCol := 24398
      mIn := 257
      mInCol := 24400
      yRingCols :=
        [((List.range 128).map (fun index => 18632 + 15 * index)),
        ((List.range 128).map (fun index => 20552 + 15 * index)),
        ((List.range 128).map (fun index => 22472 + 15 * index))]
      ctCols := [(28567, 28568), (28569, 28570), (28571, 28572)]
      rCols := [(24467, 24469)]
      sColCols := [(24497, 24499), (24501, 24503), (24505, 24507), (24509, 24511), (24513, 24515), (24517, 24519), (24521, 24523), (24525, 24527), (24529, 24531)]
      foldDigestCols := ((List.range 4).map (fun index => 28651 + 2 * index)) },
    {
      commitment := { dCol := 24401, kappaCol := 24402, dataCols := ((List.range 972).map (fun index => 3 + 15 * index)) }
      adv := none
      xActiveCols := ((List.range 270).map (fun index => 14583 + 15 * index))
      xInactiveCol := 24768
      xRows := 54
      xWidth := 257
      xRowsCol := 24403
      xWidthCol := 24404
      mIn := 257
      mInCol := 24405
      yRingCols :=
        [((List.range 128).map (fun index => 18633 + 15 * index)),
        ((List.range 128).map (fun index => 20553 + 15 * index)),
        ((List.range 128).map (fun index => 22473 + 15 * index))]
      ctCols := [(28573, 28574), (28575, 28576), (28577, 28578)]
      rCols := [(24470, 24471)]
      sColCols := [(24532, 24533), (24534, 24535), (24536, 24537), (24538, 24539), (24540, 24541), (24542, 24543), (24544, 24545), (24546, 24547), (24548, 24549)]
      foldDigestCols := ((List.range 4).map (fun index => 28659 + 1 * index)) },
    {
      commitment := { dCol := 24406, kappaCol := 24407, dataCols := ((List.range 972).map (fun index => 4 + 15 * index)) }
      adv := none
      xActiveCols := ((List.range 270).map (fun index => 14584 + 15 * index))
      xInactiveCol := 24769
      xRows := 54
      xWidth := 257
      xRowsCol := 24408
      xWidthCol := 24409
      mIn := 257
      mInCol := 24410
      yRingCols :=
        [((List.range 128).map (fun index => 18634 + 15 * index)),
        ((List.range 128).map (fun index => 20554 + 15 * index)),
        ((List.range 128).map (fun index => 22474 + 15 * index))]
      ctCols := [(28579, 28580), (28581, 28582), (28583, 28584)]
      rCols := [(24472, 24473)]
      sColCols := [(24550, 24551), (24552, 24553), (24554, 24555), (24556, 24557), (24558, 24559), (24560, 24561), (24562, 24563), (24564, 24565), (24566, 24567)]
      foldDigestCols := ((List.range 4).map (fun index => 28663 + 1 * index)) },
    {
      commitment := { dCol := 24411, kappaCol := 24412, dataCols := ((List.range 972).map (fun index => 5 + 15 * index)) }
      adv := none
      xActiveCols := ((List.range 270).map (fun index => 14585 + 15 * index))
      xInactiveCol := 24770
      xRows := 54
      xWidth := 257
      xRowsCol := 24413
      xWidthCol := 24414
      mIn := 257
      mInCol := 24415
      yRingCols :=
        [((List.range 128).map (fun index => 18635 + 15 * index)),
        ((List.range 128).map (fun index => 20555 + 15 * index)),
        ((List.range 128).map (fun index => 22475 + 15 * index))]
      ctCols := [(28585, 28586), (28587, 28588), (28589, 28590)]
      rCols := [(24474, 24475)]
      sColCols := [(24568, 24569), (24570, 24571), (24572, 24573), (24574, 24575), (24576, 24577), (24578, 24579), (24580, 24581), (24582, 24583), (24584, 24585)]
      foldDigestCols := ((List.range 4).map (fun index => 28667 + 1 * index)) },
    {
      commitment := { dCol := 24416, kappaCol := 24417, dataCols := ((List.range 972).map (fun index => 6 + 15 * index)) }
      adv := none
      xActiveCols := ((List.range 270).map (fun index => 14586 + 15 * index))
      xInactiveCol := 24771
      xRows := 54
      xWidth := 257
      xRowsCol := 24418
      xWidthCol := 24419
      mIn := 257
      mInCol := 24420
      yRingCols :=
        [((List.range 128).map (fun index => 18636 + 15 * index)),
        ((List.range 128).map (fun index => 20556 + 15 * index)),
        ((List.range 128).map (fun index => 22476 + 15 * index))]
      ctCols := [(28591, 28592), (28593, 28594), (28595, 28596)]
      rCols := [(24476, 24477)]
      sColCols := [(24586, 24587), (24588, 24589), (24590, 24591), (24592, 24593), (24594, 24595), (24596, 24597), (24598, 24599), (24600, 24601), (24602, 24603)]
      foldDigestCols := ((List.range 4).map (fun index => 28671 + 1 * index)) },
    {
      commitment := { dCol := 24421, kappaCol := 24422, dataCols := ((List.range 972).map (fun index => 7 + 15 * index)) }
      adv := none
      xActiveCols := ((List.range 270).map (fun index => 14587 + 15 * index))
      xInactiveCol := 24772
      xRows := 54
      xWidth := 257
      xRowsCol := 24423
      xWidthCol := 24424
      mIn := 257
      mInCol := 24425
      yRingCols :=
        [((List.range 128).map (fun index => 18637 + 15 * index)),
        ((List.range 128).map (fun index => 20557 + 15 * index)),
        ((List.range 128).map (fun index => 22477 + 15 * index))]
      ctCols := [(28597, 28598), (28599, 28600), (28601, 28602)]
      rCols := [(24478, 24479)]
      sColCols := [(24604, 24605), (24606, 24607), (24608, 24609), (24610, 24611), (24612, 24613), (24614, 24615), (24616, 24617), (24618, 24619), (24620, 24621)]
      foldDigestCols := ((List.range 4).map (fun index => 28675 + 1 * index)) },
    {
      commitment := { dCol := 24426, kappaCol := 24427, dataCols := ((List.range 972).map (fun index => 8 + 15 * index)) }
      adv := none
      xActiveCols := ((List.range 270).map (fun index => 14588 + 15 * index))
      xInactiveCol := 24773
      xRows := 54
      xWidth := 257
      xRowsCol := 24428
      xWidthCol := 24429
      mIn := 257
      mInCol := 24430
      yRingCols :=
        [((List.range 128).map (fun index => 18638 + 15 * index)),
        ((List.range 128).map (fun index => 20558 + 15 * index)),
        ((List.range 128).map (fun index => 22478 + 15 * index))]
      ctCols := [(28603, 28604), (28605, 28606), (28607, 28608)]
      rCols := [(24480, 24481)]
      sColCols := [(24622, 24623), (24624, 24625), (24626, 24627), (24628, 24629), (24630, 24631), (24632, 24633), (24634, 24635), (24636, 24637), (24638, 24639)]
      foldDigestCols := ((List.range 4).map (fun index => 28679 + 1 * index)) },
    {
      commitment := { dCol := 24431, kappaCol := 24432, dataCols := ((List.range 972).map (fun index => 9 + 15 * index)) }
      adv := none
      xActiveCols := ((List.range 270).map (fun index => 14589 + 15 * index))
      xInactiveCol := 24774
      xRows := 54
      xWidth := 257
      xRowsCol := 24433
      xWidthCol := 24434
      mIn := 257
      mInCol := 24435
      yRingCols :=
        [((List.range 128).map (fun index => 18639 + 15 * index)),
        ((List.range 128).map (fun index => 20559 + 15 * index)),
        ((List.range 128).map (fun index => 22479 + 15 * index))]
      ctCols := [(28609, 28610), (28611, 28612), (28613, 28614)]
      rCols := [(24482, 24483)]
      sColCols := [(24640, 24641), (24642, 24643), (24644, 24645), (24646, 24647), (24648, 24649), (24650, 24651), (24652, 24653), (24654, 24655), (24656, 24657)]
      foldDigestCols := ((List.range 4).map (fun index => 28683 + 1 * index)) },
    {
      commitment := { dCol := 24436, kappaCol := 24437, dataCols := ((List.range 972).map (fun index => 10 + 15 * index)) }
      adv := none
      xActiveCols := ((List.range 270).map (fun index => 14590 + 15 * index))
      xInactiveCol := 24775
      xRows := 54
      xWidth := 257
      xRowsCol := 24438
      xWidthCol := 24439
      mIn := 257
      mInCol := 24440
      yRingCols :=
        [((List.range 128).map (fun index => 18640 + 15 * index)),
        ((List.range 128).map (fun index => 20560 + 15 * index)),
        ((List.range 128).map (fun index => 22480 + 15 * index))]
      ctCols := [(28615, 28616), (28617, 28618), (28619, 28620)]
      rCols := [(24484, 24485)]
      sColCols := [(24658, 24659), (24660, 24661), (24662, 24663), (24664, 24665), (24666, 24667), (24668, 24669), (24670, 24671), (24672, 24673), (24674, 24675)]
      foldDigestCols := ((List.range 4).map (fun index => 28687 + 1 * index)) },
    {
      commitment := { dCol := 24441, kappaCol := 24442, dataCols := ((List.range 972).map (fun index => 11 + 15 * index)) }
      adv := none
      xActiveCols := ((List.range 270).map (fun index => 14591 + 15 * index))
      xInactiveCol := 24776
      xRows := 54
      xWidth := 257
      xRowsCol := 24443
      xWidthCol := 24444
      mIn := 257
      mInCol := 24445
      yRingCols :=
        [((List.range 128).map (fun index => 18641 + 15 * index)),
        ((List.range 128).map (fun index => 20561 + 15 * index)),
        ((List.range 128).map (fun index => 22481 + 15 * index))]
      ctCols := [(28621, 28622), (28623, 28624), (28625, 28626)]
      rCols := [(24486, 24487)]
      sColCols := [(24676, 24677), (24678, 24679), (24680, 24681), (24682, 24683), (24684, 24685), (24686, 24687), (24688, 24689), (24690, 24691), (24692, 24693)]
      foldDigestCols := ((List.range 4).map (fun index => 28691 + 1 * index)) },
    {
      commitment := { dCol := 24446, kappaCol := 24447, dataCols := ((List.range 972).map (fun index => 12 + 15 * index)) }
      adv := none
      xActiveCols := ((List.range 270).map (fun index => 14592 + 15 * index))
      xInactiveCol := 24777
      xRows := 54
      xWidth := 257
      xRowsCol := 24448
      xWidthCol := 24449
      mIn := 257
      mInCol := 24450
      yRingCols :=
        [((List.range 128).map (fun index => 18642 + 15 * index)),
        ((List.range 128).map (fun index => 20562 + 15 * index)),
        ((List.range 128).map (fun index => 22482 + 15 * index))]
      ctCols := [(28627, 28628), (28629, 28630), (28631, 28632)]
      rCols := [(24488, 24489)]
      sColCols := [(24694, 24695), (24696, 24697), (24698, 24699), (24700, 24701), (24702, 24703), (24704, 24705), (24706, 24707), (24708, 24709), (24710, 24711)]
      foldDigestCols := ((List.range 4).map (fun index => 28695 + 1 * index)) },
    {
      commitment := { dCol := 24451, kappaCol := 24452, dataCols := ((List.range 972).map (fun index => 13 + 15 * index)) }
      adv := none
      xActiveCols := ((List.range 270).map (fun index => 14593 + 15 * index))
      xInactiveCol := 24778
      xRows := 54
      xWidth := 257
      xRowsCol := 24453
      xWidthCol := 24454
      mIn := 257
      mInCol := 24455
      yRingCols :=
        [((List.range 128).map (fun index => 18643 + 15 * index)),
        ((List.range 128).map (fun index => 20563 + 15 * index)),
        ((List.range 128).map (fun index => 22483 + 15 * index))]
      ctCols := [(28633, 28634), (28635, 28636), (28637, 28638)]
      rCols := [(24490, 24491)]
      sColCols := [(24712, 24713), (24714, 24715), (24716, 24717), (24718, 24719), (24720, 24721), (24722, 24723), (24724, 24725), (24726, 24727), (24728, 24729)]
      foldDigestCols := ((List.range 4).map (fun index => 28699 + 1 * index)) },
    {
      commitment := { dCol := 24456, kappaCol := 24457, dataCols := ((List.range 972).map (fun index => 14 + 15 * index)) }
      adv := none
      xActiveCols := ((List.range 270).map (fun index => 14594 + 15 * index))
      xInactiveCol := 24779
      xRows := 54
      xWidth := 257
      xRowsCol := 24458
      xWidthCol := 24459
      mIn := 257
      mInCol := 24460
      yRingCols :=
        [((List.range 128).map (fun index => 18644 + 15 * index)),
        ((List.range 128).map (fun index => 20564 + 15 * index)),
        ((List.range 128).map (fun index => 22484 + 15 * index))]
      ctCols := [(28639, 28640), (28641, 28642), (28643, 28644)]
      rCols := [(24492, 24493)]
      sColCols := [(24730, 24731), (24732, 24733), (24734, 24735), (24736, 24737), (24738, 24739), (24740, 24741), (24742, 24743), (24744, 24745), (24746, 24747)]
      foldDigestCols := ((List.range 4).map (fun index => 28703 + 1 * index)) },
    {
      commitment := { dCol := 24461, kappaCol := 24462, dataCols := ((List.range 972).map (fun index => 15 + 15 * index)) }
      adv := none
      xActiveCols := ((List.range 270).map (fun index => 14595 + 15 * index))
      xInactiveCol := 24780
      xRows := 54
      xWidth := 257
      xRowsCol := 24463
      xWidthCol := 24464
      mIn := 257
      mInCol := 24465
      yRingCols :=
        [((List.range 128).map (fun index => 18645 + 15 * index)),
        ((List.range 128).map (fun index => 20565 + 15 * index)),
        ((List.range 128).map (fun index => 22485 + 15 * index))]
      ctCols := [(28645, 28646), (28647, 28648), (28649, 28650)]
      rCols := [(24494, 24495)]
      sColCols := [(24748, 24749), (24750, 24751), (24752, 24753), (24754, 24755), (24756, 24757), (24758, 24759), (24760, 24761), (24762, 24763), (24764, 24765)]
      foldDigestCols := ((List.range 4).map (fun index => 28707 + 1 * index)) }] }

end Nightstream.Implementation.R1CS.FPrimeFullHistoryPiDec

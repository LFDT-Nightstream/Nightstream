import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec.Schema

/-! Generated bounded active strict-PiDEC source layout. Do not hand-edit.

Owns: the proof-free Rust-exported layout record.

Does not own: layout validity, compiler semantics, acceptance, or row removal.

Emits constraints: no.

| Payload | Meaning | Authority |
|---|---|---|
| `value` | exact active source columns and trace pairs | untrusted until checked |
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec.Generated.Layout

open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec

set_option maxRecDepth 100000 in
def value : RawLayout := {
schemaVersion := 2
radix := 2
ringDimension := 54
extensionLimbs := 2
firstAllocatedColumn := 8588729
parent := {
      commitment := { dCol := 7896546, kappaCol := 7896547, dataCols := ((List.range 216).map (fun index => 7894188 + 1 * index)) }
      xActiveCols := ((List.range 270).map (fun index => 7894404 + 1 * index))
      xRows := 54
      xWidth := 5
      xRowsCol := 7896548
      xWidthCol := 7896549
      mIn := 270
      mInCol := 7896550
      yRingCols :=
        [((List.range 128).map (fun index => 7894674 + 1 * index)),
        ((List.range 128).map (fun index => 7894802 + 1 * index)),
        ((List.range 128).map (fun index => 7894930 + 1 * index)),
        ((List.range 128).map (fun index => 7895058 + 1 * index)),
        ((List.range 128).map (fun index => 7895186 + 1 * index)),
        ((List.range 128).map (fun index => 7895314 + 1 * index)),
        ((List.range 128).map (fun index => 7895442 + 1 * index)),
        ((List.range 128).map (fun index => 7895570 + 1 * index)),
        ((List.range 128).map (fun index => 7895698 + 1 * index)),
        ((List.range 128).map (fun index => 7895826 + 1 * index)),
        ((List.range 128).map (fun index => 7895954 + 1 * index)),
        ((List.range 128).map (fun index => 7896082 + 1 * index)),
        ((List.range 128).map (fun index => 7896210 + 1 * index)),
        ((List.range 128).map (fun index => 7896338 + 1 * index))]
      ctCols := [(7896466, 7896467), (7896468, 7896469), (7896470, 7896471), (7896472, 7896473), (7896474, 7896475), (7896476, 7896477), (7896478, 7896479), (7896480, 7896481), (7896482, 7896483), (7896484, 7896485), (7896486, 7896487), (7896488, 7896489), (7896490, 7896491), (7896492, 7896493)]
      rCols := [(7896494, 7896495), (7896496, 7896497), (7896498, 7896499), (7896500, 7896501), (7896502, 7896503), (7896504, 7896505), (7896506, 7896507), (7896508, 7896509), (7896510, 7896511), (7896512, 7896513), (7896514, 7896515), (7896516, 7896517), (7896518, 7896519), (7896520, 7896521), (7896522, 7896523), (7896524, 7896525), (7896526, 7896527), (7896528, 7896529), (7896530, 7896531), (7896532, 7896533), (7896534, 7896535), (7896536, 7896537), (7896538, 7896539), (7896540, 7896541)]
      foldDigestCols := ((List.range 4).map (fun index => 7896542 + 1 * index)) }
children :=
[    {
      commitment := { dCol := 7898909, kappaCol := 7898910, dataCols := ((List.range 216).map (fun index => 7896551 + 1 * index)) }
      xActiveCols := ((List.range 270).map (fun index => 7896767 + 1 * index))
      xRows := 54
      xWidth := 5
      xRowsCol := 7898911
      xWidthCol := 7898912
      mIn := 270
      mInCol := 7898913
      yRingCols :=
        [((List.range 128).map (fun index => 7897037 + 1 * index)),
        ((List.range 128).map (fun index => 7897165 + 1 * index)),
        ((List.range 128).map (fun index => 7897293 + 1 * index)),
        ((List.range 128).map (fun index => 7897421 + 1 * index)),
        ((List.range 128).map (fun index => 7897549 + 1 * index)),
        ((List.range 128).map (fun index => 7897677 + 1 * index)),
        ((List.range 128).map (fun index => 7897805 + 1 * index)),
        ((List.range 128).map (fun index => 7897933 + 1 * index)),
        ((List.range 128).map (fun index => 7898061 + 1 * index)),
        ((List.range 128).map (fun index => 7898189 + 1 * index)),
        ((List.range 128).map (fun index => 7898317 + 1 * index)),
        ((List.range 128).map (fun index => 7898445 + 1 * index)),
        ((List.range 128).map (fun index => 7898573 + 1 * index)),
        ((List.range 128).map (fun index => 7898701 + 1 * index))]
      ctCols := [(7898829, 7898830), (7898831, 7898832), (7898833, 7898834), (7898835, 7898836), (7898837, 7898838), (7898839, 7898840), (7898841, 7898842), (7898843, 7898844), (7898845, 7898846), (7898847, 7898848), (7898849, 7898850), (7898851, 7898852), (7898853, 7898854), (7898855, 7898856)]
      rCols := [(7898857, 7898858), (7898859, 7898860), (7898861, 7898862), (7898863, 7898864), (7898865, 7898866), (7898867, 7898868), (7898869, 7898870), (7898871, 7898872), (7898873, 7898874), (7898875, 7898876), (7898877, 7898878), (7898879, 7898880), (7898881, 7898882), (7898883, 7898884), (7898885, 7898886), (7898887, 7898888), (7898889, 7898890), (7898891, 7898892), (7898893, 7898894), (7898895, 7898896), (7898897, 7898898), (7898899, 7898900), (7898901, 7898902), (7898903, 7898904)]
      foldDigestCols := ((List.range 4).map (fun index => 7898905 + 1 * index)) },
    {
      commitment := { dCol := 7901272, kappaCol := 7901273, dataCols := ((List.range 216).map (fun index => 7898914 + 1 * index)) }
      xActiveCols := ((List.range 270).map (fun index => 7899130 + 1 * index))
      xRows := 54
      xWidth := 5
      xRowsCol := 7901274
      xWidthCol := 7901275
      mIn := 270
      mInCol := 7901276
      yRingCols :=
        [((List.range 128).map (fun index => 7899400 + 1 * index)),
        ((List.range 128).map (fun index => 7899528 + 1 * index)),
        ((List.range 128).map (fun index => 7899656 + 1 * index)),
        ((List.range 128).map (fun index => 7899784 + 1 * index)),
        ((List.range 128).map (fun index => 7899912 + 1 * index)),
        ((List.range 128).map (fun index => 7900040 + 1 * index)),
        ((List.range 128).map (fun index => 7900168 + 1 * index)),
        ((List.range 128).map (fun index => 7900296 + 1 * index)),
        ((List.range 128).map (fun index => 7900424 + 1 * index)),
        ((List.range 128).map (fun index => 7900552 + 1 * index)),
        ((List.range 128).map (fun index => 7900680 + 1 * index)),
        ((List.range 128).map (fun index => 7900808 + 1 * index)),
        ((List.range 128).map (fun index => 7900936 + 1 * index)),
        ((List.range 128).map (fun index => 7901064 + 1 * index))]
      ctCols := [(7901192, 7901193), (7901194, 7901195), (7901196, 7901197), (7901198, 7901199), (7901200, 7901201), (7901202, 7901203), (7901204, 7901205), (7901206, 7901207), (7901208, 7901209), (7901210, 7901211), (7901212, 7901213), (7901214, 7901215), (7901216, 7901217), (7901218, 7901219)]
      rCols := [(7901220, 7901221), (7901222, 7901223), (7901224, 7901225), (7901226, 7901227), (7901228, 7901229), (7901230, 7901231), (7901232, 7901233), (7901234, 7901235), (7901236, 7901237), (7901238, 7901239), (7901240, 7901241), (7901242, 7901243), (7901244, 7901245), (7901246, 7901247), (7901248, 7901249), (7901250, 7901251), (7901252, 7901253), (7901254, 7901255), (7901256, 7901257), (7901258, 7901259), (7901260, 7901261), (7901262, 7901263), (7901264, 7901265), (7901266, 7901267)]
      foldDigestCols := ((List.range 4).map (fun index => 7901268 + 1 * index)) },
    {
      commitment := { dCol := 7903635, kappaCol := 7903636, dataCols := ((List.range 216).map (fun index => 7901277 + 1 * index)) }
      xActiveCols := ((List.range 270).map (fun index => 7901493 + 1 * index))
      xRows := 54
      xWidth := 5
      xRowsCol := 7903637
      xWidthCol := 7903638
      mIn := 270
      mInCol := 7903639
      yRingCols :=
        [((List.range 128).map (fun index => 7901763 + 1 * index)),
        ((List.range 128).map (fun index => 7901891 + 1 * index)),
        ((List.range 128).map (fun index => 7902019 + 1 * index)),
        ((List.range 128).map (fun index => 7902147 + 1 * index)),
        ((List.range 128).map (fun index => 7902275 + 1 * index)),
        ((List.range 128).map (fun index => 7902403 + 1 * index)),
        ((List.range 128).map (fun index => 7902531 + 1 * index)),
        ((List.range 128).map (fun index => 7902659 + 1 * index)),
        ((List.range 128).map (fun index => 7902787 + 1 * index)),
        ((List.range 128).map (fun index => 7902915 + 1 * index)),
        ((List.range 128).map (fun index => 7903043 + 1 * index)),
        ((List.range 128).map (fun index => 7903171 + 1 * index)),
        ((List.range 128).map (fun index => 7903299 + 1 * index)),
        ((List.range 128).map (fun index => 7903427 + 1 * index))]
      ctCols := [(7903555, 7903556), (7903557, 7903558), (7903559, 7903560), (7903561, 7903562), (7903563, 7903564), (7903565, 7903566), (7903567, 7903568), (7903569, 7903570), (7903571, 7903572), (7903573, 7903574), (7903575, 7903576), (7903577, 7903578), (7903579, 7903580), (7903581, 7903582)]
      rCols := [(7903583, 7903584), (7903585, 7903586), (7903587, 7903588), (7903589, 7903590), (7903591, 7903592), (7903593, 7903594), (7903595, 7903596), (7903597, 7903598), (7903599, 7903600), (7903601, 7903602), (7903603, 7903604), (7903605, 7903606), (7903607, 7903608), (7903609, 7903610), (7903611, 7903612), (7903613, 7903614), (7903615, 7903616), (7903617, 7903618), (7903619, 7903620), (7903621, 7903622), (7903623, 7903624), (7903625, 7903626), (7903627, 7903628), (7903629, 7903630)]
      foldDigestCols := ((List.range 4).map (fun index => 7903631 + 1 * index)) },
    {
      commitment := { dCol := 7905998, kappaCol := 7905999, dataCols := ((List.range 216).map (fun index => 7903640 + 1 * index)) }
      xActiveCols := ((List.range 270).map (fun index => 7903856 + 1 * index))
      xRows := 54
      xWidth := 5
      xRowsCol := 7906000
      xWidthCol := 7906001
      mIn := 270
      mInCol := 7906002
      yRingCols :=
        [((List.range 128).map (fun index => 7904126 + 1 * index)),
        ((List.range 128).map (fun index => 7904254 + 1 * index)),
        ((List.range 128).map (fun index => 7904382 + 1 * index)),
        ((List.range 128).map (fun index => 7904510 + 1 * index)),
        ((List.range 128).map (fun index => 7904638 + 1 * index)),
        ((List.range 128).map (fun index => 7904766 + 1 * index)),
        ((List.range 128).map (fun index => 7904894 + 1 * index)),
        ((List.range 128).map (fun index => 7905022 + 1 * index)),
        ((List.range 128).map (fun index => 7905150 + 1 * index)),
        ((List.range 128).map (fun index => 7905278 + 1 * index)),
        ((List.range 128).map (fun index => 7905406 + 1 * index)),
        ((List.range 128).map (fun index => 7905534 + 1 * index)),
        ((List.range 128).map (fun index => 7905662 + 1 * index)),
        ((List.range 128).map (fun index => 7905790 + 1 * index))]
      ctCols := [(7905918, 7905919), (7905920, 7905921), (7905922, 7905923), (7905924, 7905925), (7905926, 7905927), (7905928, 7905929), (7905930, 7905931), (7905932, 7905933), (7905934, 7905935), (7905936, 7905937), (7905938, 7905939), (7905940, 7905941), (7905942, 7905943), (7905944, 7905945)]
      rCols := [(7905946, 7905947), (7905948, 7905949), (7905950, 7905951), (7905952, 7905953), (7905954, 7905955), (7905956, 7905957), (7905958, 7905959), (7905960, 7905961), (7905962, 7905963), (7905964, 7905965), (7905966, 7905967), (7905968, 7905969), (7905970, 7905971), (7905972, 7905973), (7905974, 7905975), (7905976, 7905977), (7905978, 7905979), (7905980, 7905981), (7905982, 7905983), (7905984, 7905985), (7905986, 7905987), (7905988, 7905989), (7905990, 7905991), (7905992, 7905993)]
      foldDigestCols := ((List.range 4).map (fun index => 7905994 + 1 * index)) },
    {
      commitment := { dCol := 7908361, kappaCol := 7908362, dataCols := ((List.range 216).map (fun index => 7906003 + 1 * index)) }
      xActiveCols := ((List.range 270).map (fun index => 7906219 + 1 * index))
      xRows := 54
      xWidth := 5
      xRowsCol := 7908363
      xWidthCol := 7908364
      mIn := 270
      mInCol := 7908365
      yRingCols :=
        [((List.range 128).map (fun index => 7906489 + 1 * index)),
        ((List.range 128).map (fun index => 7906617 + 1 * index)),
        ((List.range 128).map (fun index => 7906745 + 1 * index)),
        ((List.range 128).map (fun index => 7906873 + 1 * index)),
        ((List.range 128).map (fun index => 7907001 + 1 * index)),
        ((List.range 128).map (fun index => 7907129 + 1 * index)),
        ((List.range 128).map (fun index => 7907257 + 1 * index)),
        ((List.range 128).map (fun index => 7907385 + 1 * index)),
        ((List.range 128).map (fun index => 7907513 + 1 * index)),
        ((List.range 128).map (fun index => 7907641 + 1 * index)),
        ((List.range 128).map (fun index => 7907769 + 1 * index)),
        ((List.range 128).map (fun index => 7907897 + 1 * index)),
        ((List.range 128).map (fun index => 7908025 + 1 * index)),
        ((List.range 128).map (fun index => 7908153 + 1 * index))]
      ctCols := [(7908281, 7908282), (7908283, 7908284), (7908285, 7908286), (7908287, 7908288), (7908289, 7908290), (7908291, 7908292), (7908293, 7908294), (7908295, 7908296), (7908297, 7908298), (7908299, 7908300), (7908301, 7908302), (7908303, 7908304), (7908305, 7908306), (7908307, 7908308)]
      rCols := [(7908309, 7908310), (7908311, 7908312), (7908313, 7908314), (7908315, 7908316), (7908317, 7908318), (7908319, 7908320), (7908321, 7908322), (7908323, 7908324), (7908325, 7908326), (7908327, 7908328), (7908329, 7908330), (7908331, 7908332), (7908333, 7908334), (7908335, 7908336), (7908337, 7908338), (7908339, 7908340), (7908341, 7908342), (7908343, 7908344), (7908345, 7908346), (7908347, 7908348), (7908349, 7908350), (7908351, 7908352), (7908353, 7908354), (7908355, 7908356)]
      foldDigestCols := ((List.range 4).map (fun index => 7908357 + 1 * index)) },
    {
      commitment := { dCol := 7910724, kappaCol := 7910725, dataCols := ((List.range 216).map (fun index => 7908366 + 1 * index)) }
      xActiveCols := ((List.range 270).map (fun index => 7908582 + 1 * index))
      xRows := 54
      xWidth := 5
      xRowsCol := 7910726
      xWidthCol := 7910727
      mIn := 270
      mInCol := 7910728
      yRingCols :=
        [((List.range 128).map (fun index => 7908852 + 1 * index)),
        ((List.range 128).map (fun index => 7908980 + 1 * index)),
        ((List.range 128).map (fun index => 7909108 + 1 * index)),
        ((List.range 128).map (fun index => 7909236 + 1 * index)),
        ((List.range 128).map (fun index => 7909364 + 1 * index)),
        ((List.range 128).map (fun index => 7909492 + 1 * index)),
        ((List.range 128).map (fun index => 7909620 + 1 * index)),
        ((List.range 128).map (fun index => 7909748 + 1 * index)),
        ((List.range 128).map (fun index => 7909876 + 1 * index)),
        ((List.range 128).map (fun index => 7910004 + 1 * index)),
        ((List.range 128).map (fun index => 7910132 + 1 * index)),
        ((List.range 128).map (fun index => 7910260 + 1 * index)),
        ((List.range 128).map (fun index => 7910388 + 1 * index)),
        ((List.range 128).map (fun index => 7910516 + 1 * index))]
      ctCols := [(7910644, 7910645), (7910646, 7910647), (7910648, 7910649), (7910650, 7910651), (7910652, 7910653), (7910654, 7910655), (7910656, 7910657), (7910658, 7910659), (7910660, 7910661), (7910662, 7910663), (7910664, 7910665), (7910666, 7910667), (7910668, 7910669), (7910670, 7910671)]
      rCols := [(7910672, 7910673), (7910674, 7910675), (7910676, 7910677), (7910678, 7910679), (7910680, 7910681), (7910682, 7910683), (7910684, 7910685), (7910686, 7910687), (7910688, 7910689), (7910690, 7910691), (7910692, 7910693), (7910694, 7910695), (7910696, 7910697), (7910698, 7910699), (7910700, 7910701), (7910702, 7910703), (7910704, 7910705), (7910706, 7910707), (7910708, 7910709), (7910710, 7910711), (7910712, 7910713), (7910714, 7910715), (7910716, 7910717), (7910718, 7910719)]
      foldDigestCols := ((List.range 4).map (fun index => 7910720 + 1 * index)) },
    {
      commitment := { dCol := 7913087, kappaCol := 7913088, dataCols := ((List.range 216).map (fun index => 7910729 + 1 * index)) }
      xActiveCols := ((List.range 270).map (fun index => 7910945 + 1 * index))
      xRows := 54
      xWidth := 5
      xRowsCol := 7913089
      xWidthCol := 7913090
      mIn := 270
      mInCol := 7913091
      yRingCols :=
        [((List.range 128).map (fun index => 7911215 + 1 * index)),
        ((List.range 128).map (fun index => 7911343 + 1 * index)),
        ((List.range 128).map (fun index => 7911471 + 1 * index)),
        ((List.range 128).map (fun index => 7911599 + 1 * index)),
        ((List.range 128).map (fun index => 7911727 + 1 * index)),
        ((List.range 128).map (fun index => 7911855 + 1 * index)),
        ((List.range 128).map (fun index => 7911983 + 1 * index)),
        ((List.range 128).map (fun index => 7912111 + 1 * index)),
        ((List.range 128).map (fun index => 7912239 + 1 * index)),
        ((List.range 128).map (fun index => 7912367 + 1 * index)),
        ((List.range 128).map (fun index => 7912495 + 1 * index)),
        ((List.range 128).map (fun index => 7912623 + 1 * index)),
        ((List.range 128).map (fun index => 7912751 + 1 * index)),
        ((List.range 128).map (fun index => 7912879 + 1 * index))]
      ctCols := [(7913007, 7913008), (7913009, 7913010), (7913011, 7913012), (7913013, 7913014), (7913015, 7913016), (7913017, 7913018), (7913019, 7913020), (7913021, 7913022), (7913023, 7913024), (7913025, 7913026), (7913027, 7913028), (7913029, 7913030), (7913031, 7913032), (7913033, 7913034)]
      rCols := [(7913035, 7913036), (7913037, 7913038), (7913039, 7913040), (7913041, 7913042), (7913043, 7913044), (7913045, 7913046), (7913047, 7913048), (7913049, 7913050), (7913051, 7913052), (7913053, 7913054), (7913055, 7913056), (7913057, 7913058), (7913059, 7913060), (7913061, 7913062), (7913063, 7913064), (7913065, 7913066), (7913067, 7913068), (7913069, 7913070), (7913071, 7913072), (7913073, 7913074), (7913075, 7913076), (7913077, 7913078), (7913079, 7913080), (7913081, 7913082)]
      foldDigestCols := ((List.range 4).map (fun index => 7913083 + 1 * index)) },
    {
      commitment := { dCol := 7915450, kappaCol := 7915451, dataCols := ((List.range 216).map (fun index => 7913092 + 1 * index)) }
      xActiveCols := ((List.range 270).map (fun index => 7913308 + 1 * index))
      xRows := 54
      xWidth := 5
      xRowsCol := 7915452
      xWidthCol := 7915453
      mIn := 270
      mInCol := 7915454
      yRingCols :=
        [((List.range 128).map (fun index => 7913578 + 1 * index)),
        ((List.range 128).map (fun index => 7913706 + 1 * index)),
        ((List.range 128).map (fun index => 7913834 + 1 * index)),
        ((List.range 128).map (fun index => 7913962 + 1 * index)),
        ((List.range 128).map (fun index => 7914090 + 1 * index)),
        ((List.range 128).map (fun index => 7914218 + 1 * index)),
        ((List.range 128).map (fun index => 7914346 + 1 * index)),
        ((List.range 128).map (fun index => 7914474 + 1 * index)),
        ((List.range 128).map (fun index => 7914602 + 1 * index)),
        ((List.range 128).map (fun index => 7914730 + 1 * index)),
        ((List.range 128).map (fun index => 7914858 + 1 * index)),
        ((List.range 128).map (fun index => 7914986 + 1 * index)),
        ((List.range 128).map (fun index => 7915114 + 1 * index)),
        ((List.range 128).map (fun index => 7915242 + 1 * index))]
      ctCols := [(7915370, 7915371), (7915372, 7915373), (7915374, 7915375), (7915376, 7915377), (7915378, 7915379), (7915380, 7915381), (7915382, 7915383), (7915384, 7915385), (7915386, 7915387), (7915388, 7915389), (7915390, 7915391), (7915392, 7915393), (7915394, 7915395), (7915396, 7915397)]
      rCols := [(7915398, 7915399), (7915400, 7915401), (7915402, 7915403), (7915404, 7915405), (7915406, 7915407), (7915408, 7915409), (7915410, 7915411), (7915412, 7915413), (7915414, 7915415), (7915416, 7915417), (7915418, 7915419), (7915420, 7915421), (7915422, 7915423), (7915424, 7915425), (7915426, 7915427), (7915428, 7915429), (7915430, 7915431), (7915432, 7915433), (7915434, 7915435), (7915436, 7915437), (7915438, 7915439), (7915440, 7915441), (7915442, 7915443), (7915444, 7915445)]
      foldDigestCols := ((List.range 4).map (fun index => 7915446 + 1 * index)) },
    {
      commitment := { dCol := 7917813, kappaCol := 7917814, dataCols := ((List.range 216).map (fun index => 7915455 + 1 * index)) }
      xActiveCols := ((List.range 270).map (fun index => 7915671 + 1 * index))
      xRows := 54
      xWidth := 5
      xRowsCol := 7917815
      xWidthCol := 7917816
      mIn := 270
      mInCol := 7917817
      yRingCols :=
        [((List.range 128).map (fun index => 7915941 + 1 * index)),
        ((List.range 128).map (fun index => 7916069 + 1 * index)),
        ((List.range 128).map (fun index => 7916197 + 1 * index)),
        ((List.range 128).map (fun index => 7916325 + 1 * index)),
        ((List.range 128).map (fun index => 7916453 + 1 * index)),
        ((List.range 128).map (fun index => 7916581 + 1 * index)),
        ((List.range 128).map (fun index => 7916709 + 1 * index)),
        ((List.range 128).map (fun index => 7916837 + 1 * index)),
        ((List.range 128).map (fun index => 7916965 + 1 * index)),
        ((List.range 128).map (fun index => 7917093 + 1 * index)),
        ((List.range 128).map (fun index => 7917221 + 1 * index)),
        ((List.range 128).map (fun index => 7917349 + 1 * index)),
        ((List.range 128).map (fun index => 7917477 + 1 * index)),
        ((List.range 128).map (fun index => 7917605 + 1 * index))]
      ctCols := [(7917733, 7917734), (7917735, 7917736), (7917737, 7917738), (7917739, 7917740), (7917741, 7917742), (7917743, 7917744), (7917745, 7917746), (7917747, 7917748), (7917749, 7917750), (7917751, 7917752), (7917753, 7917754), (7917755, 7917756), (7917757, 7917758), (7917759, 7917760)]
      rCols := [(7917761, 7917762), (7917763, 7917764), (7917765, 7917766), (7917767, 7917768), (7917769, 7917770), (7917771, 7917772), (7917773, 7917774), (7917775, 7917776), (7917777, 7917778), (7917779, 7917780), (7917781, 7917782), (7917783, 7917784), (7917785, 7917786), (7917787, 7917788), (7917789, 7917790), (7917791, 7917792), (7917793, 7917794), (7917795, 7917796), (7917797, 7917798), (7917799, 7917800), (7917801, 7917802), (7917803, 7917804), (7917805, 7917806), (7917807, 7917808)]
      foldDigestCols := ((List.range 4).map (fun index => 7917809 + 1 * index)) },
    {
      commitment := { dCol := 7920176, kappaCol := 7920177, dataCols := ((List.range 216).map (fun index => 7917818 + 1 * index)) }
      xActiveCols := ((List.range 270).map (fun index => 7918034 + 1 * index))
      xRows := 54
      xWidth := 5
      xRowsCol := 7920178
      xWidthCol := 7920179
      mIn := 270
      mInCol := 7920180
      yRingCols :=
        [((List.range 128).map (fun index => 7918304 + 1 * index)),
        ((List.range 128).map (fun index => 7918432 + 1 * index)),
        ((List.range 128).map (fun index => 7918560 + 1 * index)),
        ((List.range 128).map (fun index => 7918688 + 1 * index)),
        ((List.range 128).map (fun index => 7918816 + 1 * index)),
        ((List.range 128).map (fun index => 7918944 + 1 * index)),
        ((List.range 128).map (fun index => 7919072 + 1 * index)),
        ((List.range 128).map (fun index => 7919200 + 1 * index)),
        ((List.range 128).map (fun index => 7919328 + 1 * index)),
        ((List.range 128).map (fun index => 7919456 + 1 * index)),
        ((List.range 128).map (fun index => 7919584 + 1 * index)),
        ((List.range 128).map (fun index => 7919712 + 1 * index)),
        ((List.range 128).map (fun index => 7919840 + 1 * index)),
        ((List.range 128).map (fun index => 7919968 + 1 * index))]
      ctCols := [(7920096, 7920097), (7920098, 7920099), (7920100, 7920101), (7920102, 7920103), (7920104, 7920105), (7920106, 7920107), (7920108, 7920109), (7920110, 7920111), (7920112, 7920113), (7920114, 7920115), (7920116, 7920117), (7920118, 7920119), (7920120, 7920121), (7920122, 7920123)]
      rCols := [(7920124, 7920125), (7920126, 7920127), (7920128, 7920129), (7920130, 7920131), (7920132, 7920133), (7920134, 7920135), (7920136, 7920137), (7920138, 7920139), (7920140, 7920141), (7920142, 7920143), (7920144, 7920145), (7920146, 7920147), (7920148, 7920149), (7920150, 7920151), (7920152, 7920153), (7920154, 7920155), (7920156, 7920157), (7920158, 7920159), (7920160, 7920161), (7920162, 7920163), (7920164, 7920165), (7920166, 7920167), (7920168, 7920169), (7920170, 7920171)]
      foldDigestCols := ((List.range 4).map (fun index => 7920172 + 1 * index)) },
    {
      commitment := { dCol := 7922539, kappaCol := 7922540, dataCols := ((List.range 216).map (fun index => 7920181 + 1 * index)) }
      xActiveCols := ((List.range 270).map (fun index => 7920397 + 1 * index))
      xRows := 54
      xWidth := 5
      xRowsCol := 7922541
      xWidthCol := 7922542
      mIn := 270
      mInCol := 7922543
      yRingCols :=
        [((List.range 128).map (fun index => 7920667 + 1 * index)),
        ((List.range 128).map (fun index => 7920795 + 1 * index)),
        ((List.range 128).map (fun index => 7920923 + 1 * index)),
        ((List.range 128).map (fun index => 7921051 + 1 * index)),
        ((List.range 128).map (fun index => 7921179 + 1 * index)),
        ((List.range 128).map (fun index => 7921307 + 1 * index)),
        ((List.range 128).map (fun index => 7921435 + 1 * index)),
        ((List.range 128).map (fun index => 7921563 + 1 * index)),
        ((List.range 128).map (fun index => 7921691 + 1 * index)),
        ((List.range 128).map (fun index => 7921819 + 1 * index)),
        ((List.range 128).map (fun index => 7921947 + 1 * index)),
        ((List.range 128).map (fun index => 7922075 + 1 * index)),
        ((List.range 128).map (fun index => 7922203 + 1 * index)),
        ((List.range 128).map (fun index => 7922331 + 1 * index))]
      ctCols := [(7922459, 7922460), (7922461, 7922462), (7922463, 7922464), (7922465, 7922466), (7922467, 7922468), (7922469, 7922470), (7922471, 7922472), (7922473, 7922474), (7922475, 7922476), (7922477, 7922478), (7922479, 7922480), (7922481, 7922482), (7922483, 7922484), (7922485, 7922486)]
      rCols := [(7922487, 7922488), (7922489, 7922490), (7922491, 7922492), (7922493, 7922494), (7922495, 7922496), (7922497, 7922498), (7922499, 7922500), (7922501, 7922502), (7922503, 7922504), (7922505, 7922506), (7922507, 7922508), (7922509, 7922510), (7922511, 7922512), (7922513, 7922514), (7922515, 7922516), (7922517, 7922518), (7922519, 7922520), (7922521, 7922522), (7922523, 7922524), (7922525, 7922526), (7922527, 7922528), (7922529, 7922530), (7922531, 7922532), (7922533, 7922534)]
      foldDigestCols := ((List.range 4).map (fun index => 7922535 + 1 * index)) },
    {
      commitment := { dCol := 7924902, kappaCol := 7924903, dataCols := ((List.range 216).map (fun index => 7922544 + 1 * index)) }
      xActiveCols := ((List.range 270).map (fun index => 7922760 + 1 * index))
      xRows := 54
      xWidth := 5
      xRowsCol := 7924904
      xWidthCol := 7924905
      mIn := 270
      mInCol := 7924906
      yRingCols :=
        [((List.range 128).map (fun index => 7923030 + 1 * index)),
        ((List.range 128).map (fun index => 7923158 + 1 * index)),
        ((List.range 128).map (fun index => 7923286 + 1 * index)),
        ((List.range 128).map (fun index => 7923414 + 1 * index)),
        ((List.range 128).map (fun index => 7923542 + 1 * index)),
        ((List.range 128).map (fun index => 7923670 + 1 * index)),
        ((List.range 128).map (fun index => 7923798 + 1 * index)),
        ((List.range 128).map (fun index => 7923926 + 1 * index)),
        ((List.range 128).map (fun index => 7924054 + 1 * index)),
        ((List.range 128).map (fun index => 7924182 + 1 * index)),
        ((List.range 128).map (fun index => 7924310 + 1 * index)),
        ((List.range 128).map (fun index => 7924438 + 1 * index)),
        ((List.range 128).map (fun index => 7924566 + 1 * index)),
        ((List.range 128).map (fun index => 7924694 + 1 * index))]
      ctCols := [(7924822, 7924823), (7924824, 7924825), (7924826, 7924827), (7924828, 7924829), (7924830, 7924831), (7924832, 7924833), (7924834, 7924835), (7924836, 7924837), (7924838, 7924839), (7924840, 7924841), (7924842, 7924843), (7924844, 7924845), (7924846, 7924847), (7924848, 7924849)]
      rCols := [(7924850, 7924851), (7924852, 7924853), (7924854, 7924855), (7924856, 7924857), (7924858, 7924859), (7924860, 7924861), (7924862, 7924863), (7924864, 7924865), (7924866, 7924867), (7924868, 7924869), (7924870, 7924871), (7924872, 7924873), (7924874, 7924875), (7924876, 7924877), (7924878, 7924879), (7924880, 7924881), (7924882, 7924883), (7924884, 7924885), (7924886, 7924887), (7924888, 7924889), (7924890, 7924891), (7924892, 7924893), (7924894, 7924895), (7924896, 7924897)]
      foldDigestCols := ((List.range 4).map (fun index => 7924898 + 1 * index)) },
    {
      commitment := { dCol := 7927265, kappaCol := 7927266, dataCols := ((List.range 216).map (fun index => 7924907 + 1 * index)) }
      xActiveCols := ((List.range 270).map (fun index => 7925123 + 1 * index))
      xRows := 54
      xWidth := 5
      xRowsCol := 7927267
      xWidthCol := 7927268
      mIn := 270
      mInCol := 7927269
      yRingCols :=
        [((List.range 128).map (fun index => 7925393 + 1 * index)),
        ((List.range 128).map (fun index => 7925521 + 1 * index)),
        ((List.range 128).map (fun index => 7925649 + 1 * index)),
        ((List.range 128).map (fun index => 7925777 + 1 * index)),
        ((List.range 128).map (fun index => 7925905 + 1 * index)),
        ((List.range 128).map (fun index => 7926033 + 1 * index)),
        ((List.range 128).map (fun index => 7926161 + 1 * index)),
        ((List.range 128).map (fun index => 7926289 + 1 * index)),
        ((List.range 128).map (fun index => 7926417 + 1 * index)),
        ((List.range 128).map (fun index => 7926545 + 1 * index)),
        ((List.range 128).map (fun index => 7926673 + 1 * index)),
        ((List.range 128).map (fun index => 7926801 + 1 * index)),
        ((List.range 128).map (fun index => 7926929 + 1 * index)),
        ((List.range 128).map (fun index => 7927057 + 1 * index))]
      ctCols := [(7927185, 7927186), (7927187, 7927188), (7927189, 7927190), (7927191, 7927192), (7927193, 7927194), (7927195, 7927196), (7927197, 7927198), (7927199, 7927200), (7927201, 7927202), (7927203, 7927204), (7927205, 7927206), (7927207, 7927208), (7927209, 7927210), (7927211, 7927212)]
      rCols := [(7927213, 7927214), (7927215, 7927216), (7927217, 7927218), (7927219, 7927220), (7927221, 7927222), (7927223, 7927224), (7927225, 7927226), (7927227, 7927228), (7927229, 7927230), (7927231, 7927232), (7927233, 7927234), (7927235, 7927236), (7927237, 7927238), (7927239, 7927240), (7927241, 7927242), (7927243, 7927244), (7927245, 7927246), (7927247, 7927248), (7927249, 7927250), (7927251, 7927252), (7927253, 7927254), (7927255, 7927256), (7927257, 7927258), (7927259, 7927260)]
      foldDigestCols := ((List.range 4).map (fun index => 7927261 + 1 * index)) },
    {
      commitment := { dCol := 7929628, kappaCol := 7929629, dataCols := ((List.range 216).map (fun index => 7927270 + 1 * index)) }
      xActiveCols := ((List.range 270).map (fun index => 7927486 + 1 * index))
      xRows := 54
      xWidth := 5
      xRowsCol := 7929630
      xWidthCol := 7929631
      mIn := 270
      mInCol := 7929632
      yRingCols :=
        [((List.range 128).map (fun index => 7927756 + 1 * index)),
        ((List.range 128).map (fun index => 7927884 + 1 * index)),
        ((List.range 128).map (fun index => 7928012 + 1 * index)),
        ((List.range 128).map (fun index => 7928140 + 1 * index)),
        ((List.range 128).map (fun index => 7928268 + 1 * index)),
        ((List.range 128).map (fun index => 7928396 + 1 * index)),
        ((List.range 128).map (fun index => 7928524 + 1 * index)),
        ((List.range 128).map (fun index => 7928652 + 1 * index)),
        ((List.range 128).map (fun index => 7928780 + 1 * index)),
        ((List.range 128).map (fun index => 7928908 + 1 * index)),
        ((List.range 128).map (fun index => 7929036 + 1 * index)),
        ((List.range 128).map (fun index => 7929164 + 1 * index)),
        ((List.range 128).map (fun index => 7929292 + 1 * index)),
        ((List.range 128).map (fun index => 7929420 + 1 * index))]
      ctCols := [(7929548, 7929549), (7929550, 7929551), (7929552, 7929553), (7929554, 7929555), (7929556, 7929557), (7929558, 7929559), (7929560, 7929561), (7929562, 7929563), (7929564, 7929565), (7929566, 7929567), (7929568, 7929569), (7929570, 7929571), (7929572, 7929573), (7929574, 7929575)]
      rCols := [(7929576, 7929577), (7929578, 7929579), (7929580, 7929581), (7929582, 7929583), (7929584, 7929585), (7929586, 7929587), (7929588, 7929589), (7929590, 7929591), (7929592, 7929593), (7929594, 7929595), (7929596, 7929597), (7929598, 7929599), (7929600, 7929601), (7929602, 7929603), (7929604, 7929605), (7929606, 7929607), (7929608, 7929609), (7929610, 7929611), (7929612, 7929613), (7929614, 7929615), (7929616, 7929617), (7929618, 7929619), (7929620, 7929621), (7929622, 7929623)]
      foldDigestCols := ((List.range 4).map (fun index => 7929624 + 1 * index)) },
    {
      commitment := { dCol := 7931991, kappaCol := 7931992, dataCols := ((List.range 216).map (fun index => 7929633 + 1 * index)) }
      xActiveCols := ((List.range 270).map (fun index => 7929849 + 1 * index))
      xRows := 54
      xWidth := 5
      xRowsCol := 7931993
      xWidthCol := 7931994
      mIn := 270
      mInCol := 7931995
      yRingCols :=
        [((List.range 128).map (fun index => 7930119 + 1 * index)),
        ((List.range 128).map (fun index => 7930247 + 1 * index)),
        ((List.range 128).map (fun index => 7930375 + 1 * index)),
        ((List.range 128).map (fun index => 7930503 + 1 * index)),
        ((List.range 128).map (fun index => 7930631 + 1 * index)),
        ((List.range 128).map (fun index => 7930759 + 1 * index)),
        ((List.range 128).map (fun index => 7930887 + 1 * index)),
        ((List.range 128).map (fun index => 7931015 + 1 * index)),
        ((List.range 128).map (fun index => 7931143 + 1 * index)),
        ((List.range 128).map (fun index => 7931271 + 1 * index)),
        ((List.range 128).map (fun index => 7931399 + 1 * index)),
        ((List.range 128).map (fun index => 7931527 + 1 * index)),
        ((List.range 128).map (fun index => 7931655 + 1 * index)),
        ((List.range 128).map (fun index => 7931783 + 1 * index))]
      ctCols := [(7931911, 7931912), (7931913, 7931914), (7931915, 7931916), (7931917, 7931918), (7931919, 7931920), (7931921, 7931922), (7931923, 7931924), (7931925, 7931926), (7931927, 7931928), (7931929, 7931930), (7931931, 7931932), (7931933, 7931934), (7931935, 7931936), (7931937, 7931938)]
      rCols := [(7931939, 7931940), (7931941, 7931942), (7931943, 7931944), (7931945, 7931946), (7931947, 7931948), (7931949, 7931950), (7931951, 7931952), (7931953, 7931954), (7931955, 7931956), (7931957, 7931958), (7931959, 7931960), (7931961, 7931962), (7931963, 7931964), (7931965, 7931966), (7931967, 7931968), (7931969, 7931970), (7931971, 7931972), (7931973, 7931974), (7931975, 7931976), (7931977, 7931978), (7931979, 7931980), (7931981, 7931982), (7931983, 7931984), (7931985, 7931986)]
      foldDigestCols := ((List.range 4).map (fun index => 7931987 + 1 * index)) },
    {
      commitment := { dCol := 7934354, kappaCol := 7934355, dataCols := ((List.range 216).map (fun index => 7931996 + 1 * index)) }
      xActiveCols := ((List.range 270).map (fun index => 7932212 + 1 * index))
      xRows := 54
      xWidth := 5
      xRowsCol := 7934356
      xWidthCol := 7934357
      mIn := 270
      mInCol := 7934358
      yRingCols :=
        [((List.range 128).map (fun index => 7932482 + 1 * index)),
        ((List.range 128).map (fun index => 7932610 + 1 * index)),
        ((List.range 128).map (fun index => 7932738 + 1 * index)),
        ((List.range 128).map (fun index => 7932866 + 1 * index)),
        ((List.range 128).map (fun index => 7932994 + 1 * index)),
        ((List.range 128).map (fun index => 7933122 + 1 * index)),
        ((List.range 128).map (fun index => 7933250 + 1 * index)),
        ((List.range 128).map (fun index => 7933378 + 1 * index)),
        ((List.range 128).map (fun index => 7933506 + 1 * index)),
        ((List.range 128).map (fun index => 7933634 + 1 * index)),
        ((List.range 128).map (fun index => 7933762 + 1 * index)),
        ((List.range 128).map (fun index => 7933890 + 1 * index)),
        ((List.range 128).map (fun index => 7934018 + 1 * index)),
        ((List.range 128).map (fun index => 7934146 + 1 * index))]
      ctCols := [(7934274, 7934275), (7934276, 7934277), (7934278, 7934279), (7934280, 7934281), (7934282, 7934283), (7934284, 7934285), (7934286, 7934287), (7934288, 7934289), (7934290, 7934291), (7934292, 7934293), (7934294, 7934295), (7934296, 7934297), (7934298, 7934299), (7934300, 7934301)]
      rCols := [(7934302, 7934303), (7934304, 7934305), (7934306, 7934307), (7934308, 7934309), (7934310, 7934311), (7934312, 7934313), (7934314, 7934315), (7934316, 7934317), (7934318, 7934319), (7934320, 7934321), (7934322, 7934323), (7934324, 7934325), (7934326, 7934327), (7934328, 7934329), (7934330, 7934331), (7934332, 7934333), (7934334, 7934335), (7934336, 7934337), (7934338, 7934339), (7934340, 7934341), (7934342, 7934343), (7934344, 7934345), (7934346, 7934347), (7934348, 7934349)]
      foldDigestCols := ((List.range 4).map (fun index => 7934350 + 1 * index)) }]
xSignTraces := [(8588729, 8588730), (8588731, 8588732), (8588733, 8588734), (8588735, 8588736), (8588737, 8588738), (8588739, 8588740), (8588741, 8588742), (8588743, 8588744), (8588745, 8588746), (8588747, 8588748), (8588749, 8588750), (8588751, 8588752), (8588753, 8588754), (8588755, 8588756), (8588757, 8588758), (8588759, 8588760), (8588761, 8588762), (8588763, 8588764), (8588765, 8588766), (8588767, 8588768), (8588769, 8588770), (8588771, 8588772), (8588773, 8588774), (8588775, 8588776), (8588777, 8588778), (8588779, 8588780), (8588781, 8588782), (8588783, 8588784), (8588785, 8588786), (8588787, 8588788), (8588789, 8588790), (8588791, 8588792), (8588793, 8588794), (8588795, 8588796), (8588797, 8588798), (8588799, 8588800), (8588801, 8588802), (8588803, 8588804), (8588805, 8588806), (8588807, 8588808), (8588809, 8588810), (8588811, 8588812), (8588813, 8588814), (8588815, 8588816), (8588817, 8588818), (8588819, 8588820), (8588821, 8588822), (8588823, 8588824), (8588825, 8588826), (8588827, 8588828), (8588829, 8588830), (8588831, 8588832), (8588833, 8588834), (8588835, 8588836), (8588837, 8588838), (8588839, 8588840), (8588841, 8588842), (8588843, 8588844), (8588845, 8588846), (8588847, 8588848), (8588849, 8588850), (8588851, 8588852), (8588853, 8588854), (8588855, 8588856), (8588857, 8588858), (8588859, 8588860), (8588861, 8588862), (8588863, 8588864), (8588865, 8588866), (8588867, 8588868), (8588869, 8588870), (8588871, 8588872), (8588873, 8588874), (8588875, 8588876), (8588877, 8588878), (8588879, 8588880), (8588881, 8588882), (8588883, 8588884), (8588885, 8588886), (8588887, 8588888), (8588889, 8588890), (8588891, 8588892), (8588893, 8588894), (8588895, 8588896), (8588897, 8588898), (8588899, 8588900), (8588901, 8588902), (8588903, 8588904), (8588905, 8588906), (8588907, 8588908), (8588909, 8588910), (8588911, 8588912), (8588913, 8588914), (8588915, 8588916), (8588917, 8588918), (8588919, 8588920), (8588921, 8588922), (8588923, 8588924), (8588925, 8588926), (8588927, 8588928), (8588929, 8588930), (8588931, 8588932), (8588933, 8588934), (8588935, 8588936), (8588937, 8588938), (8588939, 8588940), (8588941, 8588942), (8588943, 8588944), (8588945, 8588946), (8588947, 8588948), (8588949, 8588950), (8588951, 8588952), (8588953, 8588954), (8588955, 8588956), (8588957, 8588958), (8588959, 8588960), (8588961, 8588962), (8588963, 8588964), (8588965, 8588966), (8588967, 8588968), (8588969, 8588970), (8588971, 8588972), (8588973, 8588974), (8588975, 8588976), (8588977, 8588978), (8588979, 8588980), (8588981, 8588982), (8588983, 8588984), (8588985, 8588986), (8588987, 8588988), (8588989, 8588990), (8588991, 8588992), (8588993, 8588994), (8588995, 8588996), (8588997, 8588998), (8588999, 8589000), (8589001, 8589002), (8589003, 8589004), (8589005, 8589006), (8589007, 8589008), (8589009, 8589010), (8589011, 8589012), (8589013, 8589014), (8589015, 8589016), (8589017, 8589018), (8589019, 8589020), (8589021, 8589022), (8589023, 8589024), (8589025, 8589026), (8589027, 8589028), (8589029, 8589030), (8589031, 8589032), (8589033, 8589034), (8589035, 8589036), (8589037, 8589038), (8589039, 8589040), (8589041, 8589042), (8589043, 8589044), (8589045, 8589046), (8589047, 8589048), (8589049, 8589050), (8589051, 8589052), (8589053, 8589054), (8589055, 8589056), (8589057, 8589058), (8589059, 8589060), (8589061, 8589062), (8589063, 8589064), (8589065, 8589066), (8589067, 8589068), (8589069, 8589070), (8589071, 8589072), (8589073, 8589074), (8589075, 8589076), (8589077, 8589078), (8589079, 8589080), (8589081, 8589082), (8589083, 8589084), (8589085, 8589086), (8589087, 8589088), (8589089, 8589090), (8589091, 8589092), (8589093, 8589094), (8589095, 8589096), (8589097, 8589098), (8589099, 8589100), (8589101, 8589102), (8589103, 8589104), (8589105, 8589106), (8589107, 8589108), (8589109, 8589110), (8589111, 8589112), (8589113, 8589114), (8589115, 8589116), (8589117, 8589118), (8589119, 8589120), (8589121, 8589122), (8589123, 8589124), (8589125, 8589126), (8589127, 8589128), (8589129, 8589130), (8589131, 8589132), (8589133, 8589134), (8589135, 8589136), (8589137, 8589138), (8589139, 8589140), (8589141, 8589142), (8589143, 8589144), (8589145, 8589146), (8589147, 8589148), (8589149, 8589150), (8589151, 8589152), (8589153, 8589154), (8589155, 8589156), (8589157, 8589158), (8589159, 8589160), (8589161, 8589162), (8589163, 8589164), (8589165, 8589166), (8589167, 8589168), (8589169, 8589170), (8589171, 8589172), (8589173, 8589174), (8589175, 8589176), (8589177, 8589178), (8589179, 8589180), (8589181, 8589182), (8589183, 8589184), (8589185, 8589186), (8589187, 8589188), (8589189, 8589190), (8589191, 8589192), (8589193, 8589194), (8589195, 8589196), (8589197, 8589198), (8589199, 8589200), (8589201, 8589202), (8589203, 8589204), (8589205, 8589206), (8589207, 8589208), (8589209, 8589210), (8589211, 8589212), (8589213, 8589214), (8589215, 8589216), (8589217, 8589218), (8589219, 8589220), (8589221, 8589222), (8589223, 8589224), (8589225, 8589226), (8589227, 8589228), (8589229, 8589230), (8589231, 8589232), (8589233, 8589234), (8589235, 8589236), (8589237, 8589238), (8589239, 8589240), (8589241, 8589242), (8589243, 8589244), (8589245, 8589246), (8589247, 8589248), (8589249, 8589250), (8589251, 8589252), (8589253, 8589254), (8589255, 8589256), (8589257, 8589258), (8589259, 8589260), (8589261, 8589262), (8589263, 8589264), (8589265, 8589266), (8589267, 8589268)] }

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec.Generated.Layout

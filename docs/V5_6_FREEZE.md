# V5.6 freeze record: the immutable V5.4 authority graph

V5.6 development treats the artifacts below as read-only. Track A opens the
SQLite authority with `read_only=True`; Track B writes its rebuilt graph to a
separate database with its own checksums. Any run whose manifest does not
reproduce the hashes on this page is not comparable to the V5.6 baseline.

## Authority artifact

| Field | Value |
| --- | --- |
| Run directory | `artifacts/v5_4/full200_resume/v5_1_graph_ablation_full_20260805T103058Z` |
| SQLite authority | `graphmem.sqlite` |
| SQLite sha256 | `130cd67d58a063487b53a64b7dad743f8ea338337202b2492d92b039973944cd` |
| Size | 2,514,890,752 bytes |
| GraphMem commit | `1a0779acc71974fd4a653f3a242bfb387439241e` |
| Schema version | `graphmem-v5` |
| Config hash | `193e5a18a8600cf9edff47e6831ecde84ede1da14fe84a84f9e5a9d8a9f95013` |
| Semantic prompt hash | `75e8bf9f0fd17dbf437c8cdb44482a1107606305b8b948cf9037a53dca841364` |
| Models | `{"embedding": "Qwen/Qwen3-Embedding-0.6B", "llm": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"}` |
| Memories | 110 |
| Questions | 200 |
| Nodes / edges / evidence groups | 164,818 / 249,721 / 55,323 |
| Per-memory checksum digest | `646c6c4a46e2fff03e33d7ca14f5be53b7033d2d5c2654d6a86af493b81232bd` |

The per-memory checksum digest is the SHA-256 of the newline-joined
`memory_id:graph_version:graph_checksum` triples listed below, sorted by
memory id. Recomputing it is the cheapest way to prove a graph is untouched.

## Frozen retrieval baseline (as originally reported)

| Metric | Value |
| --- | ---: |
| turn_all_hit | 0.4950 |
| turn_recall | 0.6110 |
| candidate_turn_all_hit | 0.8800 |
| candidate_turn_recall | 0.9430 |
| session_all_hit | 0.7200 |
| evidence_tokens | 2011.3350 |
| visited_nodes | 73.8150 |
| visited_edges | 61.9300 |
| average_cold_equivalent_backbone_tokens | 214,869.29 |

> These numbers were computed against the **pre-adjudication draft**
> annotations (see below). They are recorded here only so the re-baseline in
> `V5_6_REBASELINE.md` has something to be compared against. Do not quote them.

## Gold annotation provenance (defect D0)

The frozen run consumed the draft annotation file, not the finalized asset
that ships in the repository:

| Role | Path | sha256 |
| --- | --- | --- |
| Used by the frozen run | `artifacts/v5/lme_gold_turn_merged_draft_20260804.jsonl` | `9ce371fe0a36ceac3de61f89673bc8f6dbd09ba02517644bce61ee123fde6a9f` |
| Finalized, adjudicated | `GraphMem/eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl` | `58a931e2337462013d96f97720cb74be1757caeb7290644cb9914a5f34019de6` |

`run_manifest.json` records the draft hash as
`9ce371fe0a36ceac3de61f89673bc8f6dbd09ba02517644bce61ee123fde6a9f`. Every V5.6 run must instead
pass the finalized file, and its manifest must record the finalized hash.

## Benchmark inputs

| Input | sha256 |
| --- | --- |
| `artifacts/development_sets/hard_lme_multisession_temporal_locomo_cat1_cat2_200_20260804/locomo_hard_cat1_multihop50_cat2_temporal50.json` | `99624e7de430f5766cbc9d13e814a3d1965c8e70a8730ef7f680eb6509bfae87` |
| `artifacts/development_sets/hard_lme_multisession_temporal_locomo_cat1_cat2_200_20260804/longmemeval_hard_multisession50_temporal50.json` | `566d250a6b013f384d9f7966aee5427a80f490f10e82b2de3067add85d5b828f` |
| `artifacts/v5/lme_gold_turn_merged_draft_20260804.jsonl` | `9ce371fe0a36ceac3de61f89673bc8f6dbd09ba02517644bce61ee123fde6a9f` |
| `configs/v5/v5_4_navigable.json` | `41c94e8ad49dcba5745be3343c3fd674914d6fb5e0b328a043ba588a546e002f` |

## Per-memory graph checksums

| memory_id | graph_version | node_count | edge_count | evidence_groups | graph_checksum |
| --- | ---: | ---: | ---: | ---: | --- |
| `00ca467f` | 12 | 1390 | 2130 | 486 | `df3d98bbd7478acaa2b26dd1aad924b58f1ff8fa9462ddafa1cb6ff36c986f38` |
| `09ba9854_abs` | 10 | 1621 | 2484 | 513 | `4e413ff5ad4566474e268124ab5792e291c6ac1b5da82fc1e1c4388984bf9de3` |
| `1f2b8d4f` | 41 | 1640 | 2561 | 522 | `668bfc83ae61c3fd68e35078ee968eafe39d8c6fe940fab9cac287f2f7f602eb` |
| `2311e44b` | 41 | 1509 | 2228 | 493 | `dbe780c4c740103a611dbf192be4a65a6fab212d71eccdb9e4eb909684ae97a6` |
| `2311e44b_abs` | 41 | 1525 | 2270 | 494 | `cc92c2b72b9fd50cae223452581bdb482109947182de2b58f4b5d56513b503ee` |
| `2788b940` | 9 | 1520 | 2359 | 475 | `550145db5f97439a49e96a44ca510517968a5fef18eea01c00d3b042ce1a100e` |
| `28dc39ac` | 9 | 1247 | 1960 | 431 | `53e3b224cf40d37fbafd153180173c96e26d1ed7ffcd3543d4e9be33e73398c8` |
| `2ce6a0f2` | 39 | 1496 | 2319 | 480 | `7da860740ec04cf6d6eb2eaa780499e95ab42c2008678006e24b03624a932f16` |
| `2ebe6c90` | 9 | 1414 | 2163 | 485 | `06ed85adcdd6dd77be53d3500f9c5a10657b597077201164fdb74bac2734582c` |
| `2ebe6c92` | 40 | 1554 | 2358 | 487 | `664f3baf54d740885510bc360671a01eecb1b791f3ec47ec2377fc82afd9412e` |
| `36b9f61e` | 40 | 1530 | 2385 | 490 | `f5431588728a443addb920ef29587c6bfaceec85cb1ee3a0d11eb43e1b744cd3` |
| `370a8ff4` | 40 | 1540 | 2303 | 498 | `4d5faa6e4fccd04c77bcf559f06c89ac7a5d84c3bc1692972ccde6be5150b611` |
| `3c1045c8` | 40 | 1623 | 2383 | 542 | `3aa8bba3bea726f29db18db9fecbd0d8aefd904282d9eed72302ee5009b6fd74` |
| `3fdac837` | 9 | 1369 | 2105 | 453 | `637df984c3c5285d61dca40e92c4309c24f339a1f20690f20e1faad9e1c7f362` |
| `4f54b7c9` | 9 | 1515 | 2312 | 491 | `e28afcd1bde577e8368f339c1d403d263056ca471fc6b9c97098163c37836624` |
| `60bf93ed_abs` | 40 | 1623 | 2443 | 521 | `496698e23a3cf8161a3b686552680ddce382e85dd8392b8bf78fec087a68b9cd` |
| `6e984301` | 9 | 1653 | 2494 | 513 | `3f177b8acdd08accf7b6fb0a3edf7e8596ab4a60e4f58a3f729e80ea82b9c9be` |
| `6e984302` | 9 | 1438 | 2175 | 460 | `6e9227ecc317d2e79360bdca4d386e4dffc57fe4af2b8104c748e0adaf31e405` |
| `7024f17c` | 40 | 1474 | 2295 | 483 | `a4c0d92994ebd940c3e0dafddaf68c6baff86f916e9653aee7dcd97c0edafeda` |
| `8077ef71` | 40 | 1588 | 2386 | 554 | `10c3bb3202f7c1f20683bb28b4749940165defd096f046690ce26c60eab97d4a` |
| `80ec1f4f_abs` | 9 | 1487 | 2336 | 479 | `07b4dc99054fa9073959c59715ea7f80d0230651baa747db6ab8a1a3d5dcff14` |
| `87f22b4a` | 40 | 1402 | 2152 | 474 | `cb9b6449f925a178ea681ab24795eccdcb65e8cc97bee44970b81486704f0533` |
| `88432d0a` | 40 | 1505 | 2224 | 479 | `dda7f5784c2bd9ea066d9add31d933ee2116567d2210a6867af7fe46948f1d1e` |
| `88432d0a_abs` | 9 | 1581 | 2410 | 510 | `828ef465903f3885ab31e37d2216e33c6ca0b1997837db9670a070c456f0eccf` |
| `8c18457d` | 9 | 1252 | 1989 | 434 | `edbf7dede5b8e22c80ae705a407886a01875a119618e6590e87d4bcdcb98f908` |
| `982b5123` | 40 | 1479 | 2292 | 473 | `60521d3263f5aae108434af2a912a9965403abc7ec9e986ab65fe2b86e805f2f` |
| `982b5123_abs` | 9 | 1545 | 2347 | 505 | `9bb2a5b31c0de2e27c1dd9b65d16f3af40479e3afc5aa758fb3891e190ad3d70` |
| `9a707b81` | 9 | 1658 | 2545 | 530 | `c1dc9fce8bf71b1a0e671792ee027d5b374454f57a52a3a47069ca60d398691c` |
| `9a707b82` | 9 | 1662 | 2532 | 532 | `f906da303f9d72da18c7b3f5e0d3ff8a96a9dd8e92aed92fc2905c3839df7002` |
| `9aaed6a3` | 9 | 1679 | 2545 | 541 | `5720685612323a249012cf87e841f772af35dcede98973a31465009bcf72f40b` |
| `9d25d4e0` | 9 | 1519 | 2324 | 493 | `505fd1acb384310c5ad7cd2c9bc48fa486a7246ef2861c9f7605d636d4ed56c9` |
| `a3045048` | 40 | 1808 | 2681 | 584 | `c09ddebe78b04e07f42959bf32ef93aa21b927057a752b59f8ff8e50ddf209b2` |
| `a346bb18` | 8 | 1499 | 2229 | 478 | `b10ed3a31d463fdbd07541fafddb25b00126a2b81a27977cf1ff2076126ad347` |
| `a3838d2b` | 40 | 1526 | 2283 | 484 | `145ca3e890aeeedde22aeffefb7c1aa3fe838399117c1b78e9815422cbbe1a51` |
| `a4996e51` | 9 | 1536 | 2296 | 512 | `5791f8ed979abb56b1ba1ff83abf7f0cf53c8d1cac81ec97fb9ebd62e344c3b3` |
| `a9f6b44c` | 9 | 1394 | 2106 | 476 | `170ee1a4c9d1b3a8da26e0a4c192c4b46fbce558e905fc7b43e67ff1f254b003` |
| `aae3761f` | 9 | 1490 | 2321 | 485 | `73d1685d0ce13f58cbb0700665d2386ce05e9ef27fe7e8f6c517b6320da137f1` |
| `b29f3365` | 40 | 1523 | 2329 | 494 | `6e0cd87afaea7b5935365c3f70df151c4de996e86fd412a073210e793b97766e` |
| `b3c15d39` | 9 | 1556 | 2390 | 502 | `888247676fb7d6f3d40db7d4cec3ea9e64820f7bd9e652f23d80859332d3d71e` |
| `b5ef892d` | 9 | 1575 | 2343 | 516 | `75474acc2955a9cd5baffe49dd49c9c4a47e524664dbb71363280c7e381322b8` |
| `ba358f49` | 9 | 1194 | 1784 | 409 | `323a16677213b9f3420543428b672ff7e2091f6f295035e644dfa68c28c8a212` |
| `ba358f49_abs` | 8 | 1446 | 2268 | 477 | `9f1e529e7e58be1ce413bac49b7acb9edce10742af9fcb6a951577afefa01a03` |
| `bf659f65` | 9 | 1597 | 2444 | 516 | `4560ac18c62179eaf67623ec13fa86e1c94a7a5da9b5685e35c9bd198fe66a9f` |
| `c8090214` | 8 | 1321 | 1965 | 459 | `1678c003acf67b44b38b5f225d405b5303ce3844e1cb50cf0620eb7bdf12eb59` |
| `c8090214_abs` | 40 | 1525 | 2275 | 506 | `0cde8cea0abff2720f28bb8297efbc4d1396743453a9c765dde5fe22907c1be4` |
| `cc6d1ec1` | 40 | 1420 | 2162 | 457 | `787a64fe997fe32c0b29fc2be88dd75d5c88b93b6f5a88a1b8924867b97f4c24` |
| `d3ab962e` | 9 | 1658 | 2451 | 530 | `1fd7e63a4d7c7bf98a40a78c21cc2e6ca93bcc5ed8224f2722be7d4f1508ecc7` |
| `d682f1a2` | 9 | 1491 | 2270 | 469 | `7c0ada5167f2ac6b80021630bb047d5f2b5102e6e7202f29a56d219cf8df1066` |
| `dcfa8644` | 40 | 1436 | 2213 | 470 | `4966662f5afdbb6363820d3cd90528ad29ec3021e37308e1571cc18f944f5b5d` |
| `dd2973ad` | 9 | 1545 | 2331 | 516 | `659c6b3b2976cb3324ad6d92decf2032ac4c9ef9cee81e70c8f06b5404d11f9f` |
| `e25c3b8d` | 9 | 1655 | 2524 | 530 | `4a31c24edbaddd21feabf7ca23e6ea946d94b4db066cf1a22c5598f9164e2a08` |
| `e3038f8c` | 9 | 1479 | 2274 | 458 | `54a0dd3310430f8449360eb7b9ac6fb1a171dccb891bf3f7f496496f9c8e1758` |
| `e4e14d04` | 9 | 1394 | 2165 | 461 | `3b7d00605647f0e66f39b4403117d25b45b4201cb639c214e1097f7dc1dce599` |
| `e831120c` | 9 | 1488 | 2236 | 481 | `09edc3491a30c4af839d88e997a1f1e41d9a608817c97c4aefa2b72d30f763d6` |
| `eac54add` | 9 | 1414 | 2251 | 476 | `23134caaa5ed503d3fc8993f402fb35b1dcb9a2921733bc43906ce49632d0690` |
| `edced276` | 9 | 1470 | 2207 | 482 | `a8994d7d9e9d4bc972032da8e3c0459f2982a243bfd1ce1bd3e35b6862a2e571` |
| `edced276_abs` | 9 | 1734 | 2548 | 544 | `b043a6ef6d7a6757fcfb93fc825c985e43642607017580dfe0587ce817937f19` |
| `ef9cf60a` | 9 | 1692 | 2440 | 522 | `d91337760a8c9edfcdd052bbd9ce46097dcdc31da0c6dfbac6566ca4000b3cc9` |
| `f35224e0` | 9 | 1718 | 2494 | 515 | `fd6cf8c228d4015a404bf53f15b1ca9272647b009ae9103bf103974bfcaae3fd` |
| `gpt4_15e38248` | 9 | 1376 | 2109 | 470 | `6d1e4ad7849b089e9153eb3a726dca70aadd3e2d1b3cb5ba2493a145528e6b11` |
| `gpt4_1d80365e` | 9 | 1637 | 2507 | 517 | `28515180ade2ce5e64f295567c00f989c4bf0f3abe644860f9f9c0b8995489a8` |
| `gpt4_2c50253f` | 9 | 1796 | 2645 | 591 | `eb59a8c16f7e1cc046afd7defcf172c101b1e9032a349bb34142902cb00a3514` |
| `gpt4_2f8be40d` | 9 | 1433 | 2234 | 474 | `cdc84b5ef630c6a7f526c5bade8c11dcd50c90f9ca7b038d6ab368871683f57b` |
| `gpt4_31ff4165` | 9 | 1473 | 2321 | 457 | `48be39e3bd6e010acf43b905f3bb28f19afd0b065ca0a11eede14e976ace4cb6` |
| `gpt4_372c3eed` | 9 | 1445 | 2262 | 465 | `f14a9dbe172d1c1ff1a488fa87cf19949cddcc1b6d861056f6b41c3d9f0ec809` |
| `gpt4_372c3eed_abs` | 9 | 1420 | 2172 | 490 | `f2469c0899404f0e585293fe6dbd6fc986b1b594819fe803883d2422896711b1` |
| `gpt4_468eb063` | 9 | 1481 | 2352 | 498 | `0aa7ba57f6e7ff0426ac19df9ad31ac1384c98d2e1fa307740561716a2826c3b` |
| `gpt4_4929293b` | 8 | 1414 | 2152 | 467 | `e5565acdfc7d76f1b2a262022f0b144c81765ca882673cc2e5da2a031a1b8b6f` |
| `gpt4_4fc4f797` | 9 | 1433 | 2201 | 472 | `adbe27bc1f80a8d6aa8e0d69eb4b077c052fcbaf8c34cb97e6b26759d42fdee3` |
| `gpt4_59149c77` | 9 | 1593 | 2460 | 484 | `0cef6567a104576993404b1c61ef4b37aeac1575b6cb952b3b9f4e574c9707de` |
| `gpt4_59c863d7` | 9 | 1529 | 2296 | 481 | `55f87f8182e17f9a554e32e1a0ea99ac9e3ad8e0e5b733249d3b76f232e5c1f0` |
| `gpt4_65aabe59` | 9 | 1319 | 2013 | 452 | `7725526156622abff3808ea170db10938c46ffd4760260351fef025cc435434f` |
| `gpt4_6dc9b45b` | 9 | 1792 | 2709 | 565 | `82404bb62eebb05efa3c1e81dbbf99aebcafafc5acbcffd08e12f3a583ac4002` |
| `gpt4_7a0daae1` | 9 | 1793 | 2636 | 561 | `f509813e8a30d1252bce7134f743a91ddb29af9385db189c46e5a7963bdbd146` |
| `gpt4_7abb270c` | 9 | 1504 | 2281 | 467 | `5e871c13c1677ba24547f632c778c4dfced3807f772b280c5652af8d9bc27587` |
| `gpt4_7bc6cf22` | 9 | 1633 | 2471 | 519 | `654f840af5c5e386558ebd31f784ae063ff7449a5d3a485ab81902a0ef208458` |
| `gpt4_7ca326fa` | 9 | 1598 | 2368 | 516 | `bc88f858f25b21847f7b5dfa6ea659d54df231762eaebac9aa3106191bbe8a7d` |
| `gpt4_7ddcf75f` | 8 | 1610 | 2345 | 480 | `1bb21ca5e088283f436939542a22781a1639e4fa7f658f1f18389aac28682d31` |
| `gpt4_7f6b06db` | 9 | 1705 | 2614 | 558 | `6fdbfeb437de80d58502295c8361a8b42a746bdba29d05cb958afb19b3ac8e1b` |
| `gpt4_8279ba02` | 9 | 1453 | 2268 | 489 | `678c4eba8195d0afd50c610afd6c73e13811da46048dd3e30962e56c864110a9` |
| `gpt4_8279ba03` | 9 | 1485 | 2233 | 500 | `34f15321a3699646c49ee43616e6bb61acb053559c377cf8211bc6d191e0d116` |
| `gpt4_88806d6e` | 9 | 1588 | 2449 | 493 | `6bd718e3e93aef2f1538e2a4404be8da12f98092f63dca990e1a56e6a1aa520f` |
| `gpt4_93159ced` | 9 | 1380 | 2163 | 444 | `1561daccb99df619f775bdc101dcb6725a7d99ebab7027ff26545be36542fb4f` |
| `gpt4_93159ced_abs` | 9 | 1941 | 3025 | 608 | `c990440c8aea420efde6d7d23769e70faa147187b13b598ec517ae93884ec6c7` |
| `gpt4_9a159967` | 9 | 1436 | 2174 | 470 | `a03bb4bd0ffd1ff7a480b407da467aa6dd326f4cc65334279839878895011220` |
| `gpt4_a1b77f9c` | 9 | 1513 | 2264 | 498 | `a6ab1711ac3135d2d1d9d8467507cd87fc427342ad33ce01f992bfa1d533d409` |
| `gpt4_a2d1d1f6` | 9 | 1454 | 2182 | 476 | `e5862a1b0c60f0afb6ff05306b3e1ffbeaae7ad8539a4381d105ca02879d5d1b` |
| `gpt4_a56e767c` | 9 | 1519 | 2311 | 477 | `87adf4d7ad5aafa57075f16cb8598b955287730a031ca09f78e2c190a765ec10` |
| `gpt4_ab202e7f` | 9 | 1402 | 2101 | 481 | `568b5e350d0932b72a18fb716fc138944d3d55aacfb49c6dbf7bfcc4fd1d4288` |
| `gpt4_af6db32f` | 9 | 1425 | 2162 | 476 | `d4bb16ffd8f8403225bc601c11ec92ca66996a583958ff2b0d40382e813b4bc1` |
| `gpt4_b0863698` | 9 | 1363 | 2046 | 457 | `00cdf2bac268fccd2246abc60c441fa553fd869394565ce135f485b513629907` |
| `gpt4_b5700ca9` | 9 | 1748 | 2574 | 537 | `4d499787ef229b60c16ca11a6e1e0a5ad54fe61ed6b6b665fc4b30ced6292029` |
| `gpt4_cd90e484` | 9 | 1508 | 2252 | 508 | `66ee18cc8cc28a7b2b955289aacecbec06b226301b7eff3bdbeb0cf9856496f5` |
| `gpt4_d12ceb0e` | 9 | 1406 | 2168 | 477 | `bdafd5e52db92e22abf0d5cb8248f388be8bf84a8a116ff58094fd2491cfca91` |
| `gpt4_d84a3211` | 9 | 1507 | 2279 | 495 | `652d7edaaa2e4c425858d0c4b40dbe414d51c598f4b59e3944db28bc4d4cdab3` |
| `gpt4_e05b82a6` | 9 | 1493 | 2307 | 497 | `cf8124b3e6406e72ac72140d1dffd475e655c33a98905b426f7b7573c753055b` |
| `gpt4_e072b769` | 9 | 1307 | 1952 | 444 | `1af3f5af06f5446e4a4489e419f136e0fba8f948bea3b643eaad61b5ab25d934` |
| `gpt4_e414231f` | 9 | 1575 | 2369 | 490 | `233733b8ff7af3e9bc90b426fcd6a034f0628fb16ecf3eb6ea35bcce79b06ff1` |
| `gpt4_f420262d` | 9 | 1306 | 1951 | 454 | `a98182666d50f5877e0bb02a466ec27c45f612a5d92b519b7bf408412b683087` |
| `gpt4_fa19884d` | 9 | 1737 | 2648 | 556 | `fa0a25165a70301c89a4714eb01886a0ea3c3201cad12c9b40ea4d46c0c65efe` |
| `locomo:conv-26` | 9 | 879 | 1340 | 419 | `975293c6aff3776229801408038f5602becf92a3cb215681e6cb9c363cdeefb4` |
| `locomo:conv-30` | 9 | 832 | 1319 | 369 | `bbf170982fe1bb2906a120a41950b6329d63955a05439a67af64b46aa02d2045` |
| `locomo:conv-41` | 9 | 1434 | 2046 | 663 | `2c253d1d0b8512bde90953e782a723143dfe21556a27afd306208bf46223173f` |
| `locomo:conv-42` | 42 | 1345 | 2000 | 629 | `bdc2d3a811150e02a625ac8554b206955a4969f63ced7218207541f9bdea9f94` |
| `locomo:conv-43` | 41 | 1442 | 2183 | 680 | `4e1bbc2f25e88d0506a3630b468a73aac7e9bac6e86988fc893b76f5b2629baa` |
| `locomo:conv-44` | 11 | 1490 | 2135 | 675 | `9935a55542500a6522b6eb8368a0540e284292c908017b5c34e599045ab158e3` |
| `locomo:conv-47` | 11 | 1495 | 2127 | 689 | `4832a86ae927cb9e425f26624c5024828d79219b6d8b722f7224aaa91c41a614` |
| `locomo:conv-48` | 11 | 1427 | 2047 | 681 | `de19e089aae469d17c5c68b0673cca171c842d13e8f608f27d07d07b9e223ced` |
| `locomo:conv-49` | 9 | 1111 | 1644 | 509 | `d7974ea9eba83c2d1b10811e9064a8c77c5e0d68ebe078ca31f6b0c5636d66b2` |
| `locomo:conv-50` | 11 | 1211 | 1765 | 568 | `ca7d36bdfceee255c66a7de9d6b94d358b09696c3783ae933c4e52e5b755f500` |


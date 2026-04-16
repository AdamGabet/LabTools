# New Modality Candidate Exploration

## Subject overlap counts
- 3-mod (microbiome+sleep+retina): 9656
- 5-mod (+ultrasound+nightingale): 7811
- 3-mod with >=3 dates each: 773
- 5-mod with >=3 dates each: 33

## Top KPI candidates by modality
### microbiome
- `Rep_231` | n=22670 (100.0%), std=1.4333, cv=0.485
- `Rep_3541` | n=22670 (100.0%), std=1.0707, cv=0.3207
- `Rep_467` | n=22670 (100.0%), std=1.0627, cv=0.3699
- `Rep_450` | n=22670 (100.0%), std=1.061, cv=0.3803
- `Rep_3229` | n=22670 (100.0%), std=1.0467, cv=0.3648
- `Rep_3012` | n=22670 (100.0%), std=0.98, cv=0.3461
- top2 coverage baseline(all): 12378, 02_00(all): 6391, 04_00(all): 3203

### sleep
- `total_valid_apnea_sleep_time` | n=17941 (99.6%), std=5251.7983, cv=0.2602
- `total_arousal_sleep_time` | n=17941 (99.6%), std=4404.5527, cv=0.2065
- `total_valid_arrhythmia_sleep_time` | n=17941 (99.6%), std=4328.591, cv=0.2014
- `total_sleep_time` | n=17941 (99.6%), std=4323.8404, cv=0.2005
- `total_light_sleep_time` | n=17941 (99.6%), std=3504.2207, cv=0.281
- `total_rem_sleep_time` | n=17941 (99.6%), std=1804.5207, cv=0.3565
- top2 coverage baseline(all): 10065, 02_00(all): 5595, 04_00(all): 2180

### retina_proxy
- `automorph_vein_average_width` | n=13093 (52.5%), std=1403.8356, cv=0.0717
- `automorph_artery_average_width` | n=13093 (52.5%), std=1397.1927, cv=0.0759
- `automorph_average_width` | n=13093 (52.5%), std=1101.25, cv=0.0655
- `automorph_artery_squared_curvature_tortuosity` | n=13093 (52.5%), std=18.8614, cv=0.8669
- `automorph_vein_squared_curvature_tortuosity` | n=13093 (52.5%), std=8.6209, cv=0.6916
- `automorph_squared_curvature_tortuosity` | n=13093 (52.5%), std=6.5429, cv=0.5397
- top2 coverage baseline(all): 6911, 02_00(all): 4710, 04_00(all): 1459

### ultrasound_proxy
- `r_abi` | n=22785 (91.4%), std=0.1394, cv=0.1247
- `l_abi` | n=22784 (91.4%), std=0.134, cv=0.1201
- `intima_media_th_mm_1_intima_media_thickness` | n=20183 (81.0%), std=0.1157, cv=0.1961
- `intima_media_th_1_fit` | n=10317 (41.4%), std=11.569, cv=0.1308
- `intima_media_th_mm_1_window_width` | n=10317 (41.4%), std=0.0825, cv=0.0082
- `intima_media_th_2_fit` | n=10269 (41.2%), std=11.2238, cv=0.1262
- top2 coverage baseline(all): 11910, 02_00(all): 7023, 04_00(all): 3300

### nightingale
- `GlycA` | n=7834 (100.0%), std=0.1264, cv=0.1521
- `VLDL_size` | n=7833 (100.0%), std=1.31, cv=0.0337
- `non_HDL_C` | n=7833 (100.0%), std=0.7574, cv=0.2409
- `VLDL_L` | n=7833 (100.0%), std=0.7373, cv=0.4061
- `Clinical_LDL_C` | n=7833 (100.0%), std=0.6697, cv=0.2617
- `HDL_L` | n=7833 (100.0%), std=0.5804, cv=0.213
- top2 coverage baseline(all): 6506, 02_00(all): 1327, 04_00(all): 0

## 3-Visit Feasibility (using top2 KPIs/modality)
- bench1 strict (3/3 modalities at all 3 visits): 6
- bench1 allow 1 missing modality per visit: 903
- bench2 strict (5/5 modalities at all 3 visits): 0
- bench2 allow 1 missing modality per visit: 246
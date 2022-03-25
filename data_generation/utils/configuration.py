DATASET_DIR = '/home/cabe0006/mb20_scratch/chamath/cvpr_experiments/cvpr_data'
# DATASET_DIR = '/Users/cabe0006/Projects/monash/cvpr_data'
VIDEO_CLIPS = {
    'train': ["OU10B3L2In_0", "OU10B1L1Out_0", "OU10B3L3In_0", "OU10B2L2In_0", "OU10B2L3Out_0", "OU10B2L2Out_0",
              "OU10B2L3In_0", "OU10B1L2In_0", "OU10B1L3Out_0", "OU10B1L2Out_0", "OU10B1L3In_0", "OU50B1L1In_0",
              "OU50B1L1Out_0", "OU50B1L2In_0", "OU50B2L2Out_0", "OU50B1L3In_0", "OU50B1L3Out_0", "OU50B1L2Out_0",
              "OU50B2L2In_0", "OU50B2L3In_0"],
    'val': ["OU10B3L2Out_0", "OU10B3L3Out_0", "OU10B1L1In_0", "OU50B3L2In_0", "OU50B3L2Out_0", "OU50B3L3Out_0"],
    'test': ["OU10B3L1In_0", "OU10B2L1In_0", "OU10B2L1Out_0", "OU10B3L1Out_0", "OU50B2L1In_0", "OU50B3L3In_0",
             "OU50B3L1In_0"],

    # 'train': ["OU50B1L1Out_0"],
    # 'val': ["OU10B1L1Out_0"],
    # 'test': ["OU10B1L1Out_0"],
}
VIDEO_CLIPS_TARGET = {
    'train': ['CU10L1B1In_0', 'CU10L1B1Out_0', 'CU25L1B1Out_0', 'CU25L1B1In_0',
                     'CU15L1B4In_0', 'CU15L1B4Out_0', 'CU20L1B4In_0', 'CU20L1B4Out_0',
                     'CU50L1B6In_0', 'CU50L1B6Out_0',
                     'CU10L1B5In_0', 'CU10L1B6Out_0'
                     ],
    'val': ['CU15L1B1In_0', 'CU20L1B1Out_0', 'CU10L1B4In_0', 'CU25L1B4In_0', 'CU10L1B6In_0', 'CU10L1B5Out_0'],
    'test': ['CU15L1B1Out_0', 'CU20L1B1In_0', 'CU10L1B4Out_0', 'CU25L1B4Out_0', 'CU30L1B6In_0', 'CU30L1B6Out_0']

    # 'train': ['CU50L1B6Out_0'],
    # 'val': ['CU50L1B6Out_0'],
    # 'test': ['CU50L1B6Out_0']
}

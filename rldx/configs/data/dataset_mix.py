from rldx.data.embodiment_tags import EmbodimentTag


dataset_mix = {
    "calvin": [
        {
            "dataset_name": "calvin_task_ABC_D_lerobot_0_4",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "calvin_task_ABC_D_lerobot_1_4",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "calvin_task_ABC_D_lerobot_2_4",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "calvin_task_ABC_D_lerobot_3_4",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
    ],
    "rldx1_midtrain_allex": [
        {
            "dataset_name": "real_allex",
            "mix_ratio": 0.5,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "robocurate_contiguous_seen_img_seen_instruction",
            "mix_ratio": 0.15,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "robocurate_i2i_img_novel_instruction",
            "mix_ratio": 0.25,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "robocurate_seen_img_novel_instruction",
            "mix_ratio": 0.1,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
    ],
    "rldx1_midtrain_droid": [
        {
            "dataset_name": "droid_public",
            "mix_ratio": 0.8,
            "embodiment_tag": EmbodimentTag.OXE_DROID,
        },
        {
            "dataset_name": "droid_inhouse_collected",
            "mix_ratio": 0.2,
            "embodiment_tag": EmbodimentTag.OXE_DROID,
        },
    ],
    "gr1_tabletop_1000demo": [
        {
            "dataset_name": "gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
        {
            "dataset_name": "gr1_unified.PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000",
            "mix_ratio": 1.0,
            "embodiment_tag": EmbodimentTag.GENERAL_EMBODIMENT,
        },
    ],
}

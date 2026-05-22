import torch


class SMPLXJointMapper:

    """
    Maps SMPL-X joints
    to COCO17 joint layout.
    """

    # =====================================================
    # SMPL-X -> COCO17
    # =====================================================

    COCO17_MAPPING = [

        55, # nose

        57, # left eye
        56, # right eye

        59, # left ear
        58, # right ear

        16, # left shoulder
        17, # right shoulder

        18, # left elbow
        19, # right elbow

        20, # left wrist
        21, # right wrist

        1,  # left hip
        2,  # right hip

        4,  # left knee
        5,  # right knee

        7,  # left ankle
        8   # right ankle
    ]

    @classmethod
    def smplx_to_coco17(
        cls,
        joints
    ):
        """
        joints:
            [B, J, 3]

        returns:
            [B, 17, 3]
        """

        indices = torch.tensor(
            cls.COCO17_MAPPING,
            device=joints.device
        )

        return joints[:, indices]
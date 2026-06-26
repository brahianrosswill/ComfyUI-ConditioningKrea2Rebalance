from .conditioning_rebalance import ConditioningKrea2Rebalance
from .image_edit_encode_rebalance import Krea2EditRebalance

NODE_CLASS_MAPPINGS = {
    "ConditioningKrea2Rebalance": ConditioningKrea2Rebalance,
    "Krea2EditRebalance": Krea2EditRebalance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConditioningKrea2Rebalance": "Conditioning Krea2 Rebalance",
    "Krea2EditRebalance": "Krea 2 Image Edit Rebalance",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

import os
import json
import random
import re
from typing import List, Dict, Optional, Union

class ChannelSampler:
    def __init__(
        self,
        channels: Optional[List[str]] = None,
        preprocessed_root: Optional[str] = None,
        seed: int = 42
    ):
        """
        Initialize with either a list of channel names OR a path to a preprocessed root containing channels.json.
        """
        self.rng = random.Random(seed)

        # Allow passing list directly (better for testing) or loading from file
        if channels is not None:
            self.channel_names = channels
        elif preprocessed_root is not None:
            channels_path = os.path.join(preprocessed_root, "channels.json")
            if not os.path.exists(channels_path):
                raise ValueError(f"Channels file doesn't exist at: {channels_path}")
            with open(channels_path, "r") as f:
                self.channel_names = json.load(f)
        else:
            raise ValueError("Must provide either 'channels' list or 'preprocessed_root'.")

        # Map channel name to index: {'Fp1': 0, 'Fpz': 1, ...}
        self.channels_map = {name: i for i, name in enumerate(self.channel_names)}

    def get_roi_indices(self, region_of_interest: str = None) -> List[int]:
        """
        Returns channel indices for a specific lobe/hemisphere using robust parsing.
        """
        if region_of_interest is None:
            return list(self.channels_map.values())

        roi = region_of_interest.lower()
        selected_channels = []

        for name in self.channel_names:
            name_lower = name.lower()
            
            # Extract the numeric part or 'z' to determine laterality
            # Matches strings like "3", "z", "10" at the end of the string
            match = re.search(r'(\d+|z)$', name_lower)
            suffix = match.group(0) if match else ""

            is_midline = suffix == 'z'
            # Check if number is odd (left) or even (right). 0 is treated as even usually, 
            # but in 10-20, 10 is right, 9 is left.
            is_left = False
            is_right = False
            
            if suffix.isdigit():
                num = int(suffix)
                if num % 2 != 0:
                    is_left = True
                else:
                    is_right = True

            # -- Filter Logic --
            if roi == "midline" and is_midline:
                selected_channels.append(name)
            elif roi == "left_hemisphere" and is_left:
                selected_channels.append(name)
            elif roi == "right_hemisphere" and is_right:
                selected_channels.append(name)
            
            # Lobe Logic (Standard 10-20/10-05 prefixes)
            # F = Frontal, C = Central, T = Temporal, P = Parietal, O = Occipital
            # FP = Frontal Pole (often grouped with Frontal)
            # AF = Anterior Frontal (Frontal)
            # FC = Fronto-Central (Frontal or Central depending on definition, usually Frontal)
            # CP = Centro-Parietal (Parietal)
            # PO = Parieto-Occipital (Occipital or Parietal)
            elif roi == "frontal":
                # Now includes F (Frontal), AF (Anterior Frontal), Fp (Frontopolar), and FC (Fronto-Central)
                # We still exclude FT (Fronto-Temporal) to keep it in the Temporal block
                if (name_lower.startswith(("f", "af", "fp", "fc")) 
                    and not name_lower.startswith("ft")):
                    selected_channels.append(name)

            elif roi == "central":
                # Strictly Central lines (C1, C2, Cz, etc.)
                # Note: We do not include CP or FC here to avoid overlaps with Parietal/Frontal
                if name_lower.startswith("c") and not name_lower.startswith("cp"):
                    selected_channels.append(name)

            elif roi == "temporal":
                # Includes T (Temporal), FT (Fronto-Temporal), TP (Temporo-Parietal)
                if name_lower.startswith(("t", "ft", "tp")):
                    selected_channels.append(name)

            elif roi == "parietal":
                # Includes P (Parietal) and CP (Centro-Parietal)
                if name_lower.startswith(("p", "cp")) and not name_lower.startswith("po"):
                    selected_channels.append(name)

            elif roi == "occipital":
                # Includes O (Occipital), PO (Parieto-Occipital), and I (Inion)
                if name_lower.startswith(("o", "po", "i")):
                    selected_channels.append(name)

        return [self.channels_map[k] for k in selected_channels]

    def sample_channels_per_lobe(
        self, 
        roi_idx: List[int], 
        percent_channels_per_lobe: float = None, 
        n_channels_per_lobe: int = None
    ) -> List[int]:
        
        # Determine how many to sample
        n_sample = len(roi_idx) # Default to all

        if n_channels_per_lobe is not None:
            n_sample = n_channels_per_lobe
        elif percent_channels_per_lobe is not None:
            # Ensure at least 1 channel is selected if percent > 0
            n_sample = max(1, int(len(roi_idx) * percent_channels_per_lobe))
        
        if n_sample >= len(roi_idx):
            return roi_idx
        
        roi_idx_copy = roi_idx[:]
        self.rng.shuffle(roi_idx_copy)
        return roi_idx_copy[:n_sample]

    def process_config(self, channels_config: Dict) -> List[int]:
        """
        Processes a configuration dictionary to return a final list of unique channel indices.
        """
        rois = channels_config.get("rois", [None])
        percent = channels_config.get("percent_channels_per_lobe")
        n_count = channels_config.get("n_channels_per_lobe")

        final_channels_idx = []

        for roi in rois:
            roi_idx = self.get_roi_indices(roi)
            if len(roi_idx) == 0:
                continue
            
            # Sample if config requests it
            if percent is not None or n_count is not None:
                roi_idx = self.sample_channels_per_lobe(
                    roi_idx, 
                    percent_channels_per_lobe=percent, 
                    n_channels_per_lobe=n_count
                )
            
            final_channels_idx.extend(roi_idx)

        return sorted(list(set(final_channels_idx)))


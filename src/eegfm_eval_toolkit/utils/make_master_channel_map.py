import json
import mne

def create_master_map():
    # 1. Load the superset montage (10-05 system)
    # This contains positions for >300 standard channels (Fp1, Fpz, AF1, etc.)
    montage = mne.channels.make_standard_montage('standard_1005')
    
    # 2. Get all standard channel names
    standard_names = montage.ch_names
    
    # 3. Add legacy aliases often found in TUH but not in modern 10-05
    # MNE standard_1005 uses T7/T8, but TUH often uses T3/T4.
    # We add T3/T4/T5/T6 to the valid list if they aren't there, 
    # or map them to the same ID as T7/T8 if you prefer merging.
    # Here, we will just treat them as valid unique channels to be safe.
    extras = ['T3', 'T4', 'T5', 'T6', 'A1', 'A2']
    for e in extras:
        if e not in standard_names:
            standard_names.append(e)
            
    # 4. Sort and Create Map
    standard_names.sort()
    
    # Create dictionary: "Name" -> ID
    channel_map = {name: i for i, name in enumerate(standard_names)}
    
    # 5. Save
    with open("master_channel_map.json", 'w') as f:
        json.dump(channel_map, f, indent=4)
        
    print(f"Created master map with {len(channel_map)} channels.")

if __name__ == "__main__":
    create_master_map()


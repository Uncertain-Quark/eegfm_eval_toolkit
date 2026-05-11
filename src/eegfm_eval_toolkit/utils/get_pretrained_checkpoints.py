# code for downloading the pretrained checkpoints of models
import os, sys, json 


def get_biot_checkpoints():
    checkpoint_paths = os.path.join(os.getenv("BIODL_PRETRAINED_CHECKPOINT_DIR"), "biot")
    os.makedirs("tmp/", exist_ok=True)
    os.makedirs(checkpoint_paths, exist_ok=True)
    os.system(f"git clone https://github.com/ycq091044/BIOT/ tmp/")
    os.system(f"mv tmp/pretrained-models/* {checkpoint_paths}")
    os.system(f"rm -rf tmp/")

def get_labram_checkpoints():
    checkpoint_paths = os.path.join(os.getenv("BIODL_PRETRAINED_CHECKPOINT_DIR"), "labram")
    os.makedirs("tmp/", exist_ok=True)
    os.makedirs(checkpoint_paths, exist_ok=True)

    os.system(f"git clone https://github.com/935963004/LaBraM tmp/")
    os.system(f"mv tmp/checkpoints/labram-base.pth {checkpoint_paths}/labram.pth")
    os.system(f"rm -rf tmp/")

def get_cbramod_checkpoints():
    checkpoint_paths = os.path.join(os.getenv("BIODL_PRETRAINED_CHECKPOINT_DIR"), "cbramod")
    os.makedirs("tmp/", exist_ok=True)
    os.makedirs(checkpoint_paths, exist_ok=True)

    os.system(f"hf download weighting666/CBraMod --local-dir tmp/")
    os.system(f"mv tmp/pretrained_weights.pth {checkpoint_paths}/cbramod.pth")
    os.system("rm -rf tmp/")

    

def get_csbrain_checkpoints():
    # get from google drive
    # can't download 
    pass 

if __name__ == "__main__":
    print(f"Obtaining pretrained checkpoints for BIOT")
    get_biot_checkpoints()
    print(f"Completed downloading checkpoints for BIOT")

    print(f"Obtaining LaBrAM checkpoints")
    get_labram_checkpoints()
    print(f"Completed downloading LaBrAM checkpoints")

    print(f"Obtaining CBraMod checkpoints")
    get_cbramod_checkpoints()
    print(f"Completed downloading CBraMod checkpoints")

    print(f"Obtaining CSBrain checkpoints")
    get_csbrain_checkpoints()
    print(f"Completed downloading CSBrain checkpoints")
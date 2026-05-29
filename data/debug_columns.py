import os
import sys
from datasets import load_dataset
from huggingface_hub import login
from pathlib import Path

# Adds /kaggle/working/FlashFormerv1 to your search path
parent_dir = Path(__file__).resolve().parents[1]
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))
# Force path resolution for local modules if needed
sys.path.append(os.getcwd())
from config import DataConfig  # Fixed import layout

def debug_dataset_columns():
    # Instantiate the config object to access default_factory fields
    hf_token = os.environ.get("HF_TOKEN", None)
    if not hf_token:
            print("❌ ERROR: HF_TOKEN environment variable is missing!")
            print("Please export it in your terminal first: export HF_TOKEN='your_token_here'")
            return        
    # Programmatically log into the hub to clear gated/private repository access
    print("🔑 Authenticating with Hugging Face Hub...")
    try:
        login(token=hf_token)
        print("✅ Authentication successful.")
    except Exception as e:
        print(f"❌ Authentication failed: {str(e)}")
        return
    cfg = DataConfig()
    sources = cfg.sources
    hf_token = os.environ.get("HF_TOKEN", None)
    
    print("=" * 60)
    print("🚀 STARTING DATASET COLUMN VALIDATION")
    print("=" * 60)
    
    for i, src in enumerate(sources):
        path = src["path"]
        split = src["split"]
        name = src.get("name", None)
        configured_col = src.get("text_column")
        
        print(f"\n[Dataset {i+1}/{len(sources)}] Checking: {path}")
        print(f"  - Split: {split}")
        if name:
            print(f"  - Sub-dataset/Dir: {name}")
            
        # Configure streaming kwargs matching your setup
        kwargs = {
            "streaming": True,
            "split": split,
            "token": hf_token,
        }
        if path == "bigcode/starcoderdata":
            kwargs["data_dir"] = name
        elif name is not None:
            kwargs["name"] = name

        try:
            # Fetch the stream
            ds_stream = load_dataset(path, **kwargs)
            
            # Take a peek at the first record
            preview_rows = list(ds_stream.take(1))
            
            if not preview_rows:
                print("  ❌ WARNING: Dataset loaded but returned 0 rows.")
                continue
                
            columns = list(preview_rows[0].keys())
            print(f"  ✅ Live Columns Found: {columns}")
            
            # Validate against your explicit config expectations
            if path == "openbmb/UltraInteract_sft":
                if "instruction" in columns and "response" in columns:
                    print("  🎯 Matches special case: 'instruction' & 'response' present for UltraInteract formatting.")
                else:
                    print("  ❌ ERROR: Missing 'instruction' or 'response' keys for UltraInteract.")
            else:
                if configured_col in columns:
                    print(f"  🎯 Matches your config requirement: '{configured_col}' column is present.")
                else:
                    print(f"  ❌ ERROR: Config expected '{configured_col}' but column was NOT found in the dataset.")

            # Print a small snippet of the fields you actually use
            print("  📝 Sample Extraction Peek:")
            target_cols_to_show = ["instruction", "response"] if path == "openbmb/UltraInteract_sft" else [configured_col]
            
            for col in target_cols_to_show:
                if col in columns:
                    val = preview_rows[0][col]
                    val_str = str(val)[:150].replace('\n', ' ') + "..." if len(str(val)) > 150 else str(val).replace('\n', ' ')
                    print(f"     └─ {col}: {val_str}")
                
        except Exception as e:
            print(f"  ❌ ERROR loading this source: {str(e)}")
            
    print("\n" + "=" * 60)
    print("🏁 VALIDATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    debug_dataset_columns()

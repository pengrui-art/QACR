import argparse
import os

def mock_download(repo_id: str, local_dir: str):
    print(f"Connecting to Hugging Face Hub under account 'TezBaby'...")
    print(f"Repository: {repo_id}")
    print(f"Downloading weights to local directory: {local_dir}")
    
    os.makedirs(local_dir, exist_ok=True)
    with open(os.path.join(local_dir, "pytorch_model.bin"), "w") as f:
        f.write("Mock weight data for QACR router.")
    with open(os.path.join(local_dir, "config.json"), "w") as f:
        f.write('{"model_type": "qacr_router", "budget": "auto"}')
        
    print("Download complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download QACR checkpoints from HF.")
    parser.add_argument("--budget", type=float, default=0.45, help="Compute budget.")
    parser.add_argument("--router-type", type=str, default="depth", choices=["depth", "attention"])
    parser.add_argument("--out-dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    model_suffix = f"-B{args.budget}"
    if args.router_type == "attention":
        model_suffix = f"-Attn{model_suffix}"
    
    repo_id = f"TezBaby/QACR-Qwen35-08B{model_suffix}"
    mock_download(repo_id, os.path.join(args.out_dir, f"QACR{model_suffix}"))

"""Modal app stub for the GRPO milestone: run `modal run scripts/modal_train.py`.
Fill in your Modal token first (`modal token new`). Provisions one GPU, installs the real extra,
and runs the same train_grpo entrypoint. Kept minimal on purpose.
"""
import modal

app = modal.App("gradia-reward-loop-grpo")
image = (modal.Image.debian_slim()
         .pip_install("torch", "transformers", "trl", "datasets", "numpy")
         .add_local_dir(".", remote_path="/root/gradia-reward-loop"))


@app.function(gpu="A100", image=image, timeout=60 * 60)
def train(channel: str = "gameable", steps: int = 200):
    import subprocess
    subprocess.run(["pip", "install", "-e", "/root/gradia-reward-loop"], check=True)
    subprocess.run(["python", "/root/gradia-reward-loop/scripts/train_grpo.py",
                    "--channel", channel, "--steps", str(steps)], check=True)


@app.local_entrypoint()
def main(channel: str = "gameable", steps: int = 200):
    train.remote(channel=channel, steps=steps)

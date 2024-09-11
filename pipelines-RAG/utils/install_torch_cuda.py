# import
import subprocess, sys

# main fuction
def uninstall_torch():
    try:
        # Uninstall the current version of torch
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torch"], check=True)
        subprocess.run(["echo", ">>> Uninstalled existing version of PyTorch."])
    except subprocess.CalledProcessError as e:
        subprocess.run(["echo", f">>> Error during uninstallation: {e}"])

async def install_torch_with_cuda():
    # uninstall
    uninstall_torch()

    # install
    try:
        # Define the command to install torch with CUDA 12.1
        subprocess.run(["echo", ">>> Reinstalling PyTorch with CUDA 12.1 ..."])
        command = [
            sys.executable, "-m", "pip", "install", "-q",
            "torch==2.4.0",
            "torchvision",
            "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ]
        
        # Run the pip command
        subprocess.run(command, check=True)
        subprocess.run(["echo", ">>> PyTorch with CUDA 12.1 installed successfully."])
        
    except subprocess.CalledProcessError as e:
        subprocess.run(["echo", f">>> Error during installation: {e}"])

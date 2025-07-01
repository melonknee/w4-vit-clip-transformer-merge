import torch
import torch.nn as nn
import wandb

def init_wandb(config={}):
    default_config = {
        "learning_rate": 0.2,
        "epochs": 5,
        "architecture": "Transformers",
    }
    # Start a new wandb run to track this script.
    return wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="andrewphillipo-wandb2-mlx",
        # Set the wandb project where this run will be logged.
        project="mnist-transformers",
        # Track hyperparameters and run metadata.
        config={**default_config, **config},
    )

def save_artifact(model_name, model_description, file_extension='pt', type="model"):
    artifact = wandb.Artifact(
        name=model_name,
        type=type,
        description=model_description
    )
    artifact.add_file(f"./data/{model_name}.{file_extension}")
    wandb.log_artifact(artifact)

def load_model_path(model_name):
    downloaded_model_path = wandb.use_model(model_name)
    return downloaded_model_path

def load_artifact_path(artifact_name, version="latest", file_extension='csv'):
    artifact = wandb.use_artifact(f"{artifact_name}:{version}")
    directory = artifact.download()
    return f"{directory}/{artifact_name}.{file_extension}"

def get_device_string():
    if (torch.cuda.is_available()):
        return "cuda"
    elif (torch.backends.mps.is_available()):
        return "mps"
    else:
        return "cpu"

def get_device():
    return torch.device(get_device_string())

def clones(module, N):
    return nn.ModuleList([module for _ in range(N)])

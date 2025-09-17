

import torch
from src.modules.jtwbio_encoder import JTwBio_Encoder
from src.data_modules.jtwbio_datamodule import ECGPPGDataModule
from lightning.pytorch.cli import LightningCLI


if __name__ == "__main__":

    torch.set_float32_matmul_precision('high')

    print("Starting training...")

    cli = LightningCLI(
        JTwBio_Encoder,
        ECGPPGDataModule,
        save_config_callback=None,
        # run=False
    )

    print("\n Training processes complete")


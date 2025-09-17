
import torch
from src.modules.jtwbio_tokenizer import JTwBio_Tokenzier 
from src.data_modules.jtwbio_datamodule import ECGPPGDataModule
from lightning.pytorch.cli import LightningCLI


def cli_main():

    cli = LightningCLI(
        JTwBio_Tokenzier,
        ECGPPGDataModule,
        save_config_callback=None, 
        run=False 
    )

    print("Starting training...")

    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

    print("Training finished.")
 

    # BEST_MODEL_PATH = 
    # print(f"Running test on specified checkpoint: {BEST_MODEL_PATH}")
    
    # cli.trainer.test(
    #     model=cli.model,
    #     datamodule=cli.datamodule,
    #     ckpt_path=BEST_MODEL_PATH
    # )

    # print("Testing finished.")


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    cli_main()

    
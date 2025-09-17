## CROSS-MODALITY SELF-SUPERVISION FOR UNIMODAL PHYSIOLOGICAL SIGNALS REPRESENTATION LEARNING

### How to Use
The workflow is divided into three main steps: pre-training the tokenizer, pre-training the encoder, and fine-tuning on downstream tasks.
1. Pre-train the Joint Tokenizer
Run the tokenizer pre-training script. This will learn the joint vocabulary from synchronous ECG-PPG pairs and save the trained tokenizer model.

2. Pre-train the Unimodal Encoder
Using the trained tokenizer from the previous step, pre-train the unimodal encoder. This encoder will learn to predict the joint tokens from a single modality.

`
torchrun --nproc_per_node=4 --rdzv_endpoint=localhost:29504 -m src.modules.encoder_finetuning

python3 -m app.run_jtwbio_pretraining fit --config configs/jtwbio_encoder_config.yaml  

torchrun --nproc_per_node=4 --rdzv_endpoint=localhost:29504 -m src.modules.tokenizer_finetuning

python3 -m src.app.run_jtwbio_encoder_pretraining fit --config src/configs/jtwbio_encoder_config.yaml

`


python src/train_encoder.py \
    --config configs/encoder_pretrain.yaml \
    --data_dir data/pulsedb \
    --tokenizer_path experiments/tokenizer/best_model.pt \
    --output_dir experiments/encoder
    
3. Fine-tune on a Downstream Task
Fine-tune the pre-trained unimodal encoder on a specific downstream task, such as HR estimation or ECG classification. Note that the input data for this stage is unimodal.

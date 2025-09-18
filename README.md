## CROSS-MODALITY SELF-SUPERVISION FOR UNIMODAL PHYSIOLOGICAL SIGNALS REPRESENTATION LEARNING

## Getting Started 

### Environment

```
git clone https://github.com/ximosss/CM-SSL.git
cd CM-SSL
conda create -n cmssl python=3.11
conda activate cmssl
pip install -r requirements.txt
```

### Dataset
**Pretraining dataset**

We use PulseDB as ours pretraining dataset. More info please refer to PulseDB [paper](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2022.1090854/full) and offical [Repository](https://github.com/pulselabteam/PulseDB)

**Finetuning dataset**

downstream datasets we used are as follows:

PPG part
+ [UCI](https://springernature.figshare.com/articles/dataset/UCI_dataset/20496258?backTo=%2Fcollections%2F_%2F6150390&file=38671886): The Cuff-Less Blood Pressure Estimation Dataset from the UCI Machine Learning Repository
+ [BIDMC](https://physionet.org/content/bidmc/1.0.0/): BIDMC PPG and Respiration Dataset
+ [PPG_DALIA](https://archive.ics.uci.edu/dataset/495/ppg+dalia): PPG-DaLiA contains data from 15 subjects wearing physiological and motion sensors, providing a PPG dataset for motion compensation and heart rate estimation in Daily Life Activities.

ECG part 
+ [CSN](https://physionet.org/content/ecg-arrhythmia/1.0.0/): A large scale 12-lead electrocardiogram database for arrhythmia study
+ [Physionet2017](https://physionet.org/content/challenge-2017/1.0.0/): AF Classification from a Short Single Lead ECG Recording: The PhysioNet/Computing in Cardiology Challenge 2017
+ [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/): a large publicly available electrocardiography dataset



## How to Use

The workflow is divided into three main steps: pre-training the tokenizer, pre-training the encoder, and fine-tuning on downstream tasks.

1. Pre-train the Joint Tokenizer
Run the tokenizer pre-training script. This will learn the joint vocabulary from synchronous ECG-PPG pairs and save the trained tokenizer model.
    ```
    python3 -m src.app.run_jtwbio_tokenizer_pretraining --config src/configs/jtwbio_tokenizer_config.yaml
    ```

2. Pre-train the Unimodal Encoder
Using the trained tokenizer from the previous step, pre-train the unimodal encoder. This encoder will learn to predict the joint tokens from a single modality.
    ```
    python3 -m src.app.run_jtwbio_encoder_pretraining fit --config src/configs/jtwbio_encoder_config.yaml
    ```

3. Fine-tune on a Downstream Task
Fine-tune the pre-trained unimodal encoder on a specific downstream task, such as HR estimation or ECG classification. Note that the input data for this stage is unimodal.
    ```
    torchrun --nproc_per_node=4 --rdzv_endpoint=localhost:29500 -m src.modules.encoder_finetuning
    ```
    or using bash script instead

    ```
    ./run_finetuning.sh -m "encoder" -e "ppg_dalia_rr ppg_dalia_hr bidmc_rr bidmc_hr uci_sbp uci_dbp"
    ```


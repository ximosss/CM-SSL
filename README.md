## CROSS-MODALITY SELF-SUPERVISION FOR UNIMODAL PHYSIOLOGICAL SIGNALS REPRESENTATION LEARNING

### Getting Started 

### Environment 
```
conda create -n cmssl python=3.11
conda activate cmssl
pip install -r requirements.txt
cd cmssl

```


### How to Use

The workflow is divided into three main steps: pre-training the tokenizer, pre-training the encoder, and fine-tuning on downstream tasks.
1. Pre-train the Joint Tokenizer
Run the tokenizer pre-training script. This will learn the joint vocabulary from synchronous ECG-PPG pairs and save the trained tokenizer model.

2. Pre-train the Unimodal Encoder
Using the trained tokenizer from the previous step, pre-train the unimodal encoder. This encoder will learn to predict the joint tokens from a single modality.

3. Fine-tune on a Downstream Task
Fine-tune the pre-trained unimodal encoder on a specific downstream task, such as HR estimation or ECG classification. Note that the input data for this stage is unimodal.



## Requirements
- Python 3.6+  
```pip install -r requirements.txt```
  
Please download the self-supervised checkpoints in HiDe-Prompt and put them in the /checkpoints/{checkpoint_name} directory.

## Usage
To reproduce the results mentioned in our paper, execute the training script in /training_script/{train_{dataset}_{backbone}.sh}. e.g. 
```
sh training_scripts/train_imr_lora_sup21k.sh
```

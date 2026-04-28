### 🚀 Training & Fine-tuning

This repository provides three different fine-tuning strategies. Choose the one that best fits your computational resources and task requirements. 

*(Note: If you are running these commands in a Jupyter Notebook / Kaggle environment, prepend `!` to the command, e.g., `!PYTHONPATH="." python ...`)*

#### 1. End-to-End Fine-tuning (Full Parameters)
Updates all model parameters. Achieves highest performance but requires significant GPU memory.
```bash
PYTHONPATH="." python mains_fine/main_finetune_class_iter_EndToEnd.py \
    --model vit_large_patch16_power_2_yo \
    --finetune /path/to/pretrained_weight.pth \
    --data_path_tr /path/to/train.csv \
    --data_path_val /path/to/val.csv \
    --output_dir /path/to/output \
    --log_dir /path/to/logs \
    --batch_size 2 \
    --accum_iter 8 \
    --epochs 50 \
    --lr 1e-4 \
    --weight_decay 0.05
```

#### 2. LoRA Fine-tuning (Parameter-Efficient)
Injects low-rank adapters into the attention blocks. Highly recommended for standard GPUs, offering a great balance between performance and memory usage.
```bash
PYTHONPATH="." python mains_fine/main_finetune_class_iter_lora.py \
    --model vit_large_patch16_power_2_yo \
    --finetune /path/to/pretrained_weight.pth \
    --data_path_tr /path/to/train.csv \
    --data_path_val /path/to/val.csv \
    --output_dir /path/to/output \
    --log_dir /path/to/logs \
    --batch_size 2 \
    --accum_iter 8 \
    --epochs 50 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lr 3e-4 \
    --weight_decay 0.05
```

#### 3. Linear Probing (Only Head)
Freezes the entire ViT backbone and only trains the classification head. Extremely fast and memory-efficient. Useful for rapid feature evaluation.
```bash
PYTHONPATH="." python mains_fine/main_finetune_class_iter_onlyhead.py \
    --model vit_large_patch16_power_2_yo \
    --finetune /path/to/pretrained_weight.pth \
    --data_path_tr /path/to/train.csv \
    --data_path_val /path/to/val.csv \
    --output_dir /path/to/output \
    --log_dir /path/to/logs \
    --batch_size 2 \
    --accum_iter 8 \
    --epochs 50 \
    --lr 1e-2 \
    --weight_decay 0.0
```


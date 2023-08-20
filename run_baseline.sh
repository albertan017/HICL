# Please refer to the argparse definition for parameter details
python finetune_data_token.py --output_dir ./data/ --dataset_path ./data/
python finetune.py --input_dir ./data/

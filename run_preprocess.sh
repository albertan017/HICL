python preprocess_data_token.py \
--task_name hash_database \
--tokenizer_name albertan017/hashencoder

python preprocess_extract_feature.py \
--dataset_path hash_database \
--model_name albertan017/hashencoder \
--input_dir ../finetune/data/ \
--save './hash_database_feature/

python preprocess_match.py \
--hash_file ./hash_database_feature/hash_database \
--model_name albertan017/hashencoder \
--dataset_path ../finetune/data/ \
--task_name eval-stance,eval-emotion,eval-irony,eval-offensive,eval-hate,sem21-task7-humor,sem22-task6-sarcasm \
--method _hicl


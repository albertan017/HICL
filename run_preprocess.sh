# First we tokenize the #database, --task_name refers to the name of text file
python preprocess_data_token.py \
  --task_name hash_database \
  --tokenizer_name albertan017/hashencoder

# Then we leverage #Encoder to extract features for the tweets
python preprocess_extract_feature.py \
  --dataset_path hash_database \
  --model_name albertan017/hashencoder \
  --save ./hash_database_feature/

# Finally we search for the most topic similar tweets from #database
# --hash_file is the extracted feature embeddings from previous step
# --dataset_path is the downstream benchmarks path
# --task_name is the downstream tasks
# --method is the suffix to save the search results
python preprocess_match.py \
  --hash_file ./hash_database_feature/hash_database \
  --model_name albertan017/hashencoder \
  --dataset_path ../finetune/data/ \
  --task_name eval-stance,eval-emotion,eval-irony,eval-offensive,eval-hate,sem21-task7-humor,sem22-task6-sarcasm \
  --method _hicl

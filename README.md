# HICL: Hashtag-Driven In-Context Learning for Social Media Natural Language Understanding
The official implementation of the paper HICL: Hashtag-Driven In-Context Learning for Social Media Natural Language Understanding

We aim to effectively retrieve external data and properly fine-tune bi-directional models to advance generic NLU on social media. 
We first pre-train an embedding model to help any social media post in context enriching by retrieving another relevant post; then, we insert trigger terms to fuse the enriched context for language models to refer to in semantics learning under sparsity. 
The framework can easily be plugged into various task-specific fine-tuning frameworks as external features and broadly benefits downstream social media tasks.
![pre-training of #Encoder](figure/encoder-train.png)

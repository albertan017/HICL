# HICL: Hashtag-Driven In-Context Learning for Social Media Natural Language Understanding
The official implementation of the paper HICL: Hashtag-Driven In-Context Learning for Social Media Natural Language Understanding

We aim to effectively retrieve external data and properly fine-tune bi-directional models to advance generic NLU on social media. 
We first pre-train an embedding model to help any social media post in context enriching by retrieving another relevant post; then, we insert trigger terms to fuse the enriched context for language models to refer to in semantics learning under sparsity. 
The framework can easily be plugged into various task-specific fine-tuning frameworks as external features and broadly benefits downstream social media tasks.

The workflow to pre-train #Encoder on 179M Twitter posts, each containing a hashtag. 
#Encoder was pre-trained on pairwise posts, and contrastive learning guided them to learn topic relevance via learning to identify posts with the same hashtag.
We randomly noise the hashtags to avoid trivial representation.
![Alt text](figure/encoder-train.png)

The workflow of HICL fine-tuning.
A tweet x is first encoded with \encoder{} and the output is then used to search the #Database to retrieve the most topic-related tweet x'. 
After that, x' and x are paired in concatenation and inserted with trigger terms for task-specific fine-tuning. 
Here HICL can both work for tweets with and without hashtags.
![Alt text](figure/HICL.png)

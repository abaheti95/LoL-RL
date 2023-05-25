# Left-over Lunch Reinforcement Learning (LoL-RL)
Improving Language Models with Advantage-based Offline Policy Gradients. 

paper: https://arxiv.org/abs/2305.14718
```
@misc{baheti2023improving,
      title={Improving Language Models with Advantage-based Offline Policy Gradients}, 
      author={Ashutosh Baheti and Ximing Lu and Faeze Brahman and Ronan Le Bras and Maarten Sap and Mark Riedl},
      year={2023},
      eprint={2305.14718},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Experiments
Install Packages: `pip install -r requirements.txt`

### 1 IMDB Positive Sentiment Continuation

#### 1.1 Preprocessing Sentiment/Style Transfer - rewards CoLA fluency and positive sentiment classifier
```bash
python preprocess_generation_task_and_add_rewards.py -i IMDBForSeq2Seq -o data/GEM/imdb_pos_rewarded/ -cm textattack/roberta-base-CoLA -scm lvwerra/distilbert-imdb -bs 32
```

#### 1.2 Train NLL initial checkpoint from GPT-2
```bash
python train_generation_task_with_off_policy_PG.py -i data/GEM/imdb_pos_rewarded/ -tn IMDBForSeq2Seq -m gpt2 -s saved_models/PG_GEM/imdb_pos/gpt2_nll -o final_results/GEM/imdb_pos/gpt2_nll/train_log -cm textattack/roberta-base-CoLA -scm lvwerra/distilbert-imdb -algo nll -vf 2 -e 6 -t -v_bs 32
```

#### 1.3 Train NLL + GPT-2 + all Offline-RL variations

1. NLL  
```bash
mkdir -p final_results/GEM/imdb_pos/gpt2_nll_nll/train_log  
mkdir -p saved_models/PG_GEM/imdb_pos/gpt2_nll_nll  
python train_generation_task_with_off_policy_PG.py -i data/GEM/imdb_pos_rewarded/ -tn IMDBForSeq2Seq -m saved_models/PG_GEM/imdb_pos/gpt2_nll -s saved_models/PG_GEM/imdb_pos/gpt2_nll_nll -o final_results/GEM/imdb_pos/gpt2_nll_nll/train_log -cm textattack/roberta-base-CoLA -scm lvwerra/distilbert-imdb -algo nll -vf 2 -e 3 -bs 16 -as 1 -v_bs 32 -t -ev_b
```
2. weighted behavior cloning (pg)
```bash
mkdir -p final_results/GEM/imdb_pos/gpt2_nll_pg/train_log
mkdir -p saved_models/PG_GEM/imdb_pos/gpt2_nll_pg
python train_generation_task_with_off_policy_PG.py -i data/GEM/imdb_pos_rewarded/ -tn IMDBForSeq2Seq -m saved_models/PG_GEM/imdb_pos/gpt2_nll -s saved_models/PG_GEM/imdb_pos/gpt2_nll_pg -o final_results/GEM/imdb_pos/gpt2_nll_pg/train_log -cm textattack/roberta-base-CoLA -scm lvwerra/distilbert-imdb -algo pg -ts -vf 2 -e 3 -bs 16 -as 1 -v_bs 32 -t -ev_b
```
3. Reward LoL-RL
```bash
mkdir -p final_results/GEM/imdb_pos/gpt2_nll_offline_pg_seq_clip_sample/train_log
mkdir -p saved_models/PG_GEM/imdb_pos/gpt2_nll_offline_pg_seq_clip_sample
python train_generation_task_with_off_policy_PG.py -i data/GEM/imdb_pos_rewarded/ -tn IMDBForSeq2Seq -m saved_models/PG_GEM/imdb_pos/gpt2_nll -s saved_models/PG_GEM/imdb_pos/gpt2_nll_offline_pg_seq_clip_sample -o final_results/GEM/imdb_pos/gpt2_nll_offline_pg_seq_clip_sample/train_log -cm textattack/roberta-base-CoLA -scm lvwerra/distilbert-imdb -algo offline_pg_seq -c 0.2 -ts -vf 2 -e 3 -bs 16 -as 1 -v_bs 32 -t -ev_b
```
4. Advantage LoL-RL
```bash
mkdir -p final_results/GEM/imdb_pos/gpt2_nll_offline_a2c_clip_sample/train_log
mkdir -p saved_models/PG_GEM/imdb_pos/gpt2_nll_offline_a2c_clip_sample
python train_generation_task_with_off_policy_PG.py -i data/GEM/imdb_pos_rewarded/ -tn IMDBForSeq2Seq -m saved_models/PG_GEM/imdb_pos/gpt2_nll -s saved_models/PG_GEM/imdb_pos/gpt2_nll_offline_a2c_clip_sample -o final_results/GEM/imdb_pos/gpt2_nll_offline_a2c_clip_sample/train_log -cm textattack/roberta-base-CoLA -scm lvwerra/distilbert-imdb -algo offline_a2c -c 0.2 -ts -vf 2 -e 3 -bs 16 -as 1 -v_bs 32 -t -ev_b
```

#### 1.4 Train NLL + lvwerra/gpt2-imdb + all Offline-RL variations

1. NLL  
```bash
mkdir -p final_results/GEM/imdb_pos/lvwerra_gpt2_imdb_nll/train_log  
mkdir -p saved_models/PG_GEM/imdb_pos/lvwerra_gpt2_imdb_nll  
python train_generation_task_with_off_policy_PG.py -i data/GEM/imdb_pos_rewarded/ -tn IMDBForSeq2Seq -m lvwerra/gpt2-imdb -s saved_models/PG_GEM/imdb_pos/lvwerra_gpt2_imdb_nll -o final_results/GEM/imdb_pos/lvwerra_gpt2_imdb_nll/train_log -cm textattack/roberta-base-CoLA -scm lvwerra/distilbert-imdb -algo nll -vf 2 -e 3 -bs 16 -as 1 -v_bs 32 -t -ev_b
```
2. weighted behavior cloning (pg)
```bash
mkdir -p final_results/GEM/imdb_pos/lvwerra_gpt2_imdb_pg/train_log
mkdir -p saved_models/PG_GEM/imdb_pos/lvwerra_gpt2_imdb_pg
python train_generation_task_with_off_policy_PG.py -i data/GEM/imdb_pos_rewarded/ -tn IMDBForSeq2Seq -m lvwerra/gpt2-imdb -s saved_models/PG_GEM/imdb_pos/lvwerra_gpt2_imdb_pg -o final_results/GEM/imdb_pos/lvwerra_gpt2_imdb_pg/train_log -cm textattack/roberta-base-CoLA -scm lvwerra/distilbert-imdb -algo pg -ts -vf 2 -e 3 -bs 16 -as 1 -v_bs 32 -t -ev_b
```
3. Reward LoL-RL
```bash
mkdir -p final_results/GEM/imdb_pos/lvwerra_gpt2_imdb_offline_pg_seq_clip_sample/train_log
mkdir -p saved_models/PG_GEM/imdb_pos/lvwerra_gpt2_imdb_offline_pg_seq_clip_sample
python train_generation_task_with_off_policy_PG.py -i data/GEM/imdb_pos_rewarded/ -tn IMDBForSeq2Seq -m lvwerra/gpt2-imdb -s saved_models/PG_GEM/imdb_pos/lvwerra_gpt2_imdb_offline_pg_seq_clip_sample -o final_results/GEM/imdb_pos/lvwerra_gpt2_imdb_offline_pg_seq_clip_sample/train_log -cm textattack/roberta-base-CoLA -scm lvwerra/distilbert-imdb -algo offline_pg_seq -c 0.2 -ts -vf 2 -e 3 -bs 16 -as 1 -v_bs 32 -t -ev_b
```
4. Advantage LoL-RL
```bash
mkdir -p final_results/GEM/imdb_pos/lvwerra_gpt2_imdb_offline_a2c_clip_sample/train_log
mkdir -p saved_models/PG_GEM/imdb_pos/lvwerra_gpt2_imdb_offline_a2c_clip_sample
python train_generation_task_with_off_policy_PG.py -i data/GEM/imdb_pos_rewarded/ -tn IMDBForSeq2Seq -m lvwerra/gpt2-imdb -s saved_models/PG_GEM/imdb_pos/lvwerra_gpt2_imdb_offline_a2c_clip_sample -o final_results/GEM/imdb_pos/lvwerra_gpt2_imdb_offline_a2c_clip_sample/train_log -cm textattack/roberta-base-CoLA -scm lvwerra/distilbert-imdb -algo offline_a2c -c 0.2 -ts -vf 2 -e 3 -bs 16 -as 1 -v_bs 32 -t -ev_b
```

#### 1.5 Aggregate IMDBForSeq2Seq results
`python aggregate_generation_task_results.py -bmps "{'gpt2_nll': True, 'lvwerra_gpt2_imdb': True}" -tn IMDBForSeq2Seq -o final_results/GEM/imdb_pos_final_results.csv`

### 2 Commonsense Transformer - COMET  

[Download](https://storage.googleapis.com/ai2-mosaic-public/projects/symbolic-knowledge-decoding/symbolic-knowledge-distillation.tar.gz) SKD data and save in `data/symbolic_knowledge_distillation/`

#### 2.1 Convert pretrained COMET critic from original paper to huggingface format
`python convert_keras_roberta_to_huggingface.py`
> Saved the final classifer as RobertaModel, Tokenizer and Custom Classification Head with specific activations at `saved_models/comet_critic_keras_to_pytorch`.   
> Classifier is saved in "custom_roberta_classification_head.pt" file within the folder  
> Initialize the Classification head as follows: RobertaClassificationHead(1024, 512, 1)  
> Total Diff between comet_critic_pred and pytorch_critic_pred for 50 instances: 1.8537044525146484e-05

#### 2.2 ATOMIC-COMET preprocessing and reward extraction
`python preprocess_comet_and_add_rewards.py -i data/symbolic_knowledge_distillation/downloaded -it data/symbolic_knowledge_distillation/atomic2020/atomic2020_data-feb2021/ -ccm saved_models/comet_critic_keras_to_pytorch -o data/GEM/comet_rewarded/ -bs 32`

#### 2.3 comet-distill + all Offline-RL variations

1. NLL
```bash
mkdir -p final_results/GEM/comet/comet_distill_nll/train_log
mkdir -p saved_models/PG_GEM/comet/comet_distill_nll
python train_generation_task_with_off_policy_PG.py -i data/GEM/comet_rewarded/ -tn COMET -m data/symbolic_knowledge_distillation/downloaded/comet-distill/ -s saved_models/PG_GEM/comet/comet_distill_nll -o final_results/GEM/comet/comet_distill_nll/train_log -mt data/symbolic_knowledge_distillation/downloaded/comet-distill-tokenizer/ -ccm saved_models/comet_critic_keras_to_pytorch -ml 30 -algo nll -vf 16 -e 1 -bs 16 -as 1 -v_bs 32 -t -ev_b
```
2. weighted behavior cloning (pg)
```bash
mkdir -p final_results/GEM/comet/comet_distill_pg/train_log
mkdir -p saved_models/PG_GEM/comet/comet_distill_pg
python train_generation_task_with_off_policy_PG.py -i data/GEM/comet_rewarded/ -tn COMET -m data/symbolic_knowledge_distillation/downloaded/comet-distill/ -s saved_models/PG_GEM/comet/comet_distill_pg -o final_results/GEM/comet/comet_distill_pg/train_log -mt data/symbolic_knowledge_distillation/downloaded/comet-distill-tokenizer/ -ccm saved_models/comet_critic_keras_to_pytorch -ml 30 -algo pg -ts -vf 16 -e 1 -bs 16 -as 1 -v_bs 32 -t -ev_b
```
3. Reward LoL-RL
```bash
mkdir -p final_results/GEM/comet/comet_distill_offline_pg_seq_clip_sample/train_log
mkdir -p saved_models/PG_GEM/comet/comet_distill_offline_pg_seq_clip_sample
python train_generation_task_with_off_policy_PG.py -i data/GEM/comet_rewarded/ -tn COMET -m data/symbolic_knowledge_distillation/downloaded/comet-distill/ -s saved_models/PG_GEM/comet/comet_distill_offline_pg_seq_clip_sample -o final_results/GEM/comet/comet_distill_offline_pg_seq_clip_sample/train_log -mt data/symbolic_knowledge_distillation/downloaded/comet-distill-tokenizer/ -ccm saved_models/comet_critic_keras_to_pytorch -ml 30 -algo offline_pg_seq -c 0.2 -ts -vf 16 -e 1 -bs 16 -as 1 -v_bs 32 -t -ev_b
```
4. Advantage LoL-RL
```bash
mkdir -p final_results/GEM/comet/comet_distill_offline_a2c_clip_sample/train_log
mkdir -p saved_models/PG_GEM/comet/comet_distill_offline_a2c_clip_sample
python train_generation_task_with_off_policy_PG.py -i data/GEM/comet_rewarded/ -tn COMET -m data/symbolic_knowledge_distillation/downloaded/comet-distill/ -s saved_models/PG_GEM/comet/comet_distill_offline_a2c_clip_sample -o final_results/GEM/comet/comet_distill_offline_a2c_clip_sample/train_log -mt data/symbolic_knowledge_distillation/downloaded/comet-distill-tokenizer/ -ccm saved_models/comet_critic_keras_to_pytorch -ml 30 -algo offline_a2c -c 0.2 -ts -vf 16 -e 1 -bs 16 -as 1 -v_bs 32 -t -ev_b
```

#### 2.4 Aggregate COMET results
`python aggregate_generation_task_results.py -bmps "{'comet_distill': True}" -tn COMET -o final_results/GEM/comet_final_results.csv`


### 3 Reddit positive and negative comment generation task

#### 3.1 Download and preprocess Reddit Comment Scores data
Download the upvoted and downvoted reddit comment pairs from: https://www.kaggle.com/code/danofer/reddit-comments-scores-nlp/input

Positive comments score 10 percentile: [66.0, 72.0, 79.0, 88.0, 100.0, 116.0, 139.0, 174.0, 236.0, 385.0, 9582.0]  
Negative comments score 10 percentile: [-2946.0, -25.0, -18.0, -14.0, -12.0, -10.0, -9.0, -8.0, -8.0, -7.0, -6.0]  

Also [download the toxichat classifiers](https://mega.nz/file/ANhEWDiA#ky-f6HNfmgM4-QVpNv_-z5cN1yf4d0Ml6PAEWHnQVCg) and save them in `saved_models`  

`python preprocess_reddit_comment_scores_and_add_rewards.py -i data/reddit_comment_scores_kaggle/ -m microsoft/DialoGPT-medium -cm textattack/roberta-base-CoLA -ucm microsoft/DialogRPT-updown -dcm microsoft/DialogRPT-depth -om saved_models/DGPT_medium_OC_S_and_SBF_offensive_e2 -o data/reddit_comment_scores_kaggle/preprocessed `

#### 3.2 Train DialoGPT-medium NLL on Reddit Positive and Reddit Negative
`python train_generation_task_with_off_policy_PG.py -i data/reddit_comment_scores_kaggle/preprocessed -tn reddit_pos -m microsoft/DialoGPT-medium -s saved_models/PG_GEM/reddit_pos/dgpt_nll -o final_results/GEM/reddit_pos/dgpt_nll/train_log -cm textattack/roberta-base-CoLA -ucm microsoft/DialogRPT-updown -dcm microsoft/DialogRPT-depth -om saved_models/DGPT_medium_OC_S_and_SBF_offensive_e2 -algo nll -vf 2 -e 6 -t -bs 8 -as 2 -v_bs 32`

`python train_generation_task_with_off_policy_PG.py -i data/reddit_comment_scores_kaggle/preprocessed -tn reddit_neg -m microsoft/DialoGPT-medium -s saved_models/PG_GEM/reddit_neg/dgpt_nll -o final_results/GEM/reddit_neg/dgpt_nll/train_log -cm textattack/roberta-base-CoLA -ucm microsoft/DialogRPT-updown -dcm microsoft/DialogRPT-depth -om saved_models/DGPT_medium_OC_S_and_SBF_offensive_e2 -algo nll -vf 2 -e 6 -t -bs 8 -as 2 -v_bs 32`

#### 3.3 Train DialoGPT-medium NLL + all Offline RL variations on Reddit Positive and Reddit Negative 

##### 3.3.1 Reddit Positive (Upvoted Comments)
1. NLL
```bash
mkdir -p final_results/GEM/reddit_pos/dgpt_nll_nll/train_log
mkdir -p saved_models/PG_GEM/reddit_pos/dgpt_nll_nll
python train_generation_task_with_off_policy_PG.py -i data/reddit_comment_scores_kaggle/preprocessed -tn reddit_pos -m saved_models/PG_GEM/reddit_pos/dgpt_nll -s saved_models/PG_GEM/reddit_pos/dgpt_nll_nll -o final_results/GEM/reddit_pos/dgpt_nll_nll/train_log -cm textattack/roberta-base-CoLA -ucm microsoft/DialogRPT-updown -dcm microsoft/DialogRPT-depth -om saved_models/DGPT_medium_OC_S_and_SBF_offensive_e2 -algo nll -vf 2 -e 3 -bs 8 -as 2 -v_bs 32 -t -ev_b
```
2. weighted behavior cloning (pg)
```bash
mkdir -p final_results/GEM/reddit_pos/dgpt_nll_pg/train_log
mkdir -p saved_models/PG_GEM/reddit_pos/dgpt_nll_pg
python train_generation_task_with_off_policy_PG.py -i data/reddit_comment_scores_kaggle/preprocessed -tn reddit_pos -m saved_models/PG_GEM/reddit_pos/dgpt_nll -s saved_models/PG_GEM/reddit_pos/dgpt_nll_pg -o final_results/GEM/reddit_pos/dgpt_nll_pg/train_log -cm textattack/roberta-base-CoLA -ucm microsoft/DialogRPT-updown -dcm microsoft/DialogRPT-depth -om saved_models/DGPT_medium_OC_S_and_SBF_offensive_e2 -algo pg -ts -vf 2 -e 3 -bs 8 -as 2 -v_bs 32 -t -ev_b
```
3. Reward LoL-RL
```bash
mkdir -p final_results/GEM/reddit_pos/dgpt_nll_offline_pg_seq_clip_sample/train_log
mkdir -p saved_models/PG_GEM/reddit_pos/dgpt_nll_offline_pg_seq_clip_sample
python train_generation_task_with_off_policy_PG.py -i data/reddit_comment_scores_kaggle/preprocessed -tn reddit_pos -m saved_models/PG_GEM/reddit_pos/dgpt_nll -s saved_models/PG_GEM/reddit_pos/dgpt_nll_offline_pg_seq_clip_sample -o final_results/GEM/reddit_pos/dgpt_nll_offline_pg_seq_clip_sample/train_log -cm textattack/roberta-base-CoLA -ucm microsoft/DialogRPT-updown -dcm microsoft/DialogRPT-depth -om saved_models/DGPT_medium_OC_S_and_SBF_offensive_e2 -algo offline_pg_seq -c 0.2 -ts -vf 2 -e 3 -bs 8 -as 2 -v_bs 32 -t -ev_b
```
4. Advantage LoL-RL
```bash
mkdir -p final_results/GEM/reddit_pos/dgpt_nll_offline_a2c_clip_sample/train_log
mkdir -p saved_models/PG_GEM/reddit_pos/dgpt_nll_offline_a2c_clip_sample
python train_generation_task_with_off_policy_PG.py -i data/reddit_comment_scores_kaggle/preprocessed -tn reddit_pos -m saved_models/PG_GEM/reddit_pos/dgpt_nll -s saved_models/PG_GEM/reddit_pos/dgpt_nll_offline_a2c_clip_sample -o final_results/GEM/reddit_pos/dgpt_nll_offline_a2c_clip_sample/train_log -cm textattack/roberta-base-CoLA -ucm microsoft/DialogRPT-updown -dcm microsoft/DialogRPT-depth -om saved_models/DGPT_medium_OC_S_and_SBF_offensive_e2 -algo offline_a2c -c 0.2 -ts -vf 2 -e 3 -bs 8 -as 2 -v_bs 32 -t -ev_b
```
##### 3.3.2 Reddit Negative (Downvoted Comments)
1. NLL
```bash
mkdir -p final_results/GEM/reddit_neg/dgpt_nll_nll/train_log
mkdir -p saved_models/PG_GEM/reddit_neg/dgpt_nll_nll
python train_generation_task_with_off_policy_PG.py -i data/reddit_comment_scores_kaggle/preprocessed -tn reddit_neg -m saved_models/PG_GEM/reddit_neg/dgpt_nll -s saved_models/PG_GEM/reddit_neg/dgpt_nll_nll -o final_results/GEM/reddit_neg/dgpt_nll_nll/train_log -cm textattack/roberta-base-CoLA -ucm microsoft/DialogRPT-updown -dcm microsoft/DialogRPT-depth -om saved_models/DGPT_medium_OC_S_and_SBF_offensive_e2 -algo nll -vf 2 -e 3 -bs 8 -as 2 -v_bs 32 -t -ev_b
```
2. weighted behavior cloning (pg)
```bash
mkdir -p final_results/GEM/reddit_neg/dgpt_nll_pg/train_log
mkdir -p saved_models/PG_GEM/reddit_neg/dgpt_nll_pg
python train_generation_task_with_off_policy_PG.py -i data/reddit_comment_scores_kaggle/preprocessed -tn reddit_neg -m saved_models/PG_GEM/reddit_neg/dgpt_nll -s saved_models/PG_GEM/reddit_neg/dgpt_nll_pg -o final_results/GEM/reddit_neg/dgpt_nll_pg/train_log -cm textattack/roberta-base-CoLA -ucm microsoft/DialogRPT-updown -dcm microsoft/DialogRPT-depth -om saved_models/DGPT_medium_OC_S_and_SBF_offensive_e2 -algo pg -ts -vf 2 -e 3 -bs 8 -as 2 -v_bs 32 -t -ev_b
```
3. Reward LoL-RL
```bash
mkdir -p final_results/GEM/reddit_neg/dgpt_nll_offline_pg_seq_clip_sample/train_log
mkdir -p saved_models/PG_GEM/reddit_neg/dgpt_nll_offline_pg_seq_clip_sample
python train_generation_task_with_off_policy_PG.py -i data/reddit_comment_scores_kaggle/preprocessed -tn reddit_neg -m saved_models/PG_GEM/reddit_neg/dgpt_nll -s saved_models/PG_GEM/reddit_neg/dgpt_nll_offline_pg_seq_clip_sample -o final_results/GEM/reddit_neg/dgpt_nll_offline_pg_seq_clip_sample/train_log -cm textattack/roberta-base-CoLA -ucm microsoft/DialogRPT-updown -dcm microsoft/DialogRPT-depth -om saved_models/DGPT_medium_OC_S_and_SBF_offensive_e2 -algo offline_pg_seq -c 0.2 -ts -vf 2 -e 3 -bs 8 -as 2 -v_bs 32 -t -ev_b
```
4. Advantage LoL-RL
```bash
mkdir -p final_results/GEM/reddit_neg/dgpt_nll_offline_a2c_clip_sample/train_log
mkdir -p saved_models/PG_GEM/reddit_neg/dgpt_nll_offline_a2c_clip_sample
python train_generation_task_with_off_policy_PG.py -i data/reddit_comment_scores_kaggle/preprocessed -tn reddit_neg -m saved_models/PG_GEM/reddit_neg/dgpt_nll -s saved_models/PG_GEM/reddit_neg/dgpt_nll_offline_a2c_clip_sample -o final_results/GEM/reddit_neg/dgpt_nll_offline_a2c_clip_sample/train_log -cm textattack/roberta-base-CoLA -ucm microsoft/DialogRPT-updown -dcm microsoft/DialogRPT-depth -om saved_models/DGPT_medium_OC_S_and_SBF_offensive_e2 -algo offline_a2c -c 0.2 -ts -vf 2 -e 3 -bs 8 -as 2 -v_bs 32 -t -ev_b
```

#### 3.4 Aggregate Reddit POS and NEG results
```bash
python aggregate_generation_task_results.py -bmps "{'dgpt_nll': True}" -tn reddit_pos -o final_results/GEM/reddit_pos_final_results.csv
python aggregate_generation_task_results.py -bmps "{'dgpt_nll': True}" -tn reddit_neg -o final_results/GEM/reddit_neg_final_results.csv
```

### 4 Wizard of Wikipedia

#### 4.1 WOW and FaithDail Test set preprocessing
`python preprocess_wow_and_add_rewards.py -i data/wow -o data/wow/preprocessed_and_rewarded/ -m microsoft/DialoGPT-medium -cm textattack/roberta-base-CoLA -fcm McGill-NLP/roberta-large-faithcritic -dcm microsoft/DialogRPT-depth -bs 32`

#### 4.2 Train DialoGPT-medium NLL on WOW
`python train_generation_task_with_off_policy_PG.py -i data/wow/preprocessed_and_rewarded/ -tn WOW -m microsoft/DialoGPT-medium -s saved_models/PG_GEM/wow/dgpt_nll -o final_results/GEM/wow/dgpt_nll/train_log -cm textattack/roberta-base-CoLA -fcm McGill-NLP/roberta-large-faithcritic -dcm microsoft/DialogRPT-depth -algo nll -vf 2 -e 6 -t -bs 8 -as 2 -v_bs 32`

#### 4.3 Train DialoGPT-medium NLL + all Offline RL variations on WOW 
1. NLL
```bash
mkdir -p final_results/GEM/wow/dgpt_nll_nll/train_log
mkdir -p saved_models/PG_GEM/wow/dgpt_nll_nll
python train_generation_task_with_off_policy_PG.py -i data/wow/preprocessed_and_rewarded/ -tn WOW -m saved_models/PG_GEM/wow/dgpt_nll -s saved_models/PG_GEM/wow/dgpt_nll_nll -o final_results/GEM/wow/dgpt_nll_nll/train_log -cm textattack/roberta-base-CoLA -fcm McGill-NLP/roberta-large-faithcritic -dcm microsoft/DialogRPT-depth -algo nll -vf 2 -e 3 -bs 8 -as 2 -v_bs 32 -t -ev_b
```
2. weighted behavior cloning (pg)
```bash
mkdir -p final_results/GEM/wow/dgpt_nll_pg/train_log
mkdir -p saved_models/PG_GEM/wow/dgpt_nll_pg
python train_generation_task_with_off_policy_PG.py -i data/wow/preprocessed_and_rewarded/ -tn WOW -m saved_models/PG_GEM/wow/dgpt_nll -s saved_models/PG_GEM/wow/dgpt_nll_pg -o final_results/GEM/wow/dgpt_nll_pg/train_log -cm textattack/roberta-base-CoLA -fcm McGill-NLP/roberta-large-faithcritic -dcm microsoft/DialogRPT-depth -algo pg -vf 2 -e 3 -bs 8 -as 2 -v_bs 32 -t -ev_b
```
3. Reward LoL-RL
```bash
mkdir -p final_results/GEM/wow/dgpt_nll_offline_pg_seq_clip_sample/train_log
mkdir -p saved_models/PG_GEM/wow/dgpt_nll_offline_pg_seq_clip_sample
python train_generation_task_with_off_policy_PG.py -i data/wow/preprocessed_and_rewarded/ -tn WOW -m saved_models/PG_GEM/wow/dgpt_nll -s saved_models/PG_GEM/wow/dgpt_nll_offline_pg_seq_clip_sample -o final_results/GEM/wow/dgpt_nll_offline_pg_seq_clip_sample/train_log -cm textattack/roberta-base-CoLA -fcm McGill-NLP/roberta-large-faithcritic -dcm microsoft/DialogRPT-depth -algo offline_pg_seq -c 0.2 -ts -vf 2 -e 3 -bs 8 -as 2 -v_bs 32 -t -ev_b
```
4. Advantage LoL-RL
```bash
mkdir -p final_results/GEM/wow/dgpt_nll_offline_a2c_clip_sample/train_log
mkdir -p saved_models/PG_GEM/wow/dgpt_nll_offline_a2c_clip_sample
python train_generation_task_with_off_policy_PG.py -i data/wow/preprocessed_and_rewarded/ -tn WOW -m saved_models/PG_GEM/wow/dgpt_nll -s saved_models/PG_GEM/wow/dgpt_nll_offline_a2c_clip_sample -o final_results/GEM/wow/dgpt_nll_offline_a2c_clip_sample/train_log -cm textattack/roberta-base-CoLA -fcm McGill-NLP/roberta-large-faithcritic -dcm microsoft/DialogRPT-depth -algo offline_a2c -c 0.2 -ts -vf 2 -e 3 -bs 8 -as 2 -v_bs 32 -t -ev_b
```

#### 4.3 Aggregate WOW results

`python aggregate_generation_task_results.py -bmps "{'dgpt_nll': True}" -tn WOW -o final_results/GEM/wow_final_results.csv`

# Outputs

## Text Generation
- ```0S_100T_all_models_202345.parquet```: Zero-Shot - 100 tweets batch - all six models used - generated on 05/04/2023. Contains:
    - ```id```: tweet id for later reference
    - ```full_text```: original tweet
    - ```{model_name}```: model output
- ```0S_100T_bloom_gpt4all_llama7b_202344.parquet```: Zero-Shot - 100 tweets batch - GPT4All, Llama 7B, BLOOM - generated on 04/04/2023. Contains:
    - ```id```: tweet id for later reference
    - ```full_text```: original tweet
    - ```{model_name}```: model output
# **msc-thesis-llm-clustering**

## **Aim of this study**
TODO

---

## **TODOs**
- Adjust Llama 7B and 13B params to stop output generation.
- Explore different prompts, system messages and contexts

---

## **Instructions**
To download tweets and to run the models create a virtual env and install the requirements with 

```pip install -r requirements.txt```

### Initial Configurations
Create all the environment variables needed and set them. Look for variable names in the ```config.py``` file. Set these two variables from within the ```config.py```file:
- ```RAW_DATA_DIR``` - where raw datasets will be stored
- ```DATA_DIR``` - where clean and ready to use datasets will be stored
### Download Tweets
Use ```dataset.ipynb``` notebook. Set Twitter API keys (be sure to check out twitter api and developer docs) by setting:

- ```TWTR_BEARER_TOKEN```
- ```TWTR_API```
- ```TWTR_API_SECRET```
- ```TWTR_ACCESS_TOKEN```
- ```TWTR_ACCESS_TOKEN_SECRET```

Use pre-configured contexts (apparel, cars and beauty) or change them as needed.
Run the notebook, being careful about Twitter api rate limits. The datasets will be stored in the ```RAW_DATA_DIR``` as ```.parquet``` files.

Data cleaning and initial preparation can be done using this notebook too. The output will be stored in the ```DATA_DIR``` directory. 

### Build DB
Use ```bots/tweetdb.py```. Script to downlaod and store last *n* tweets from previous 7 days. Change ```DOMAINS``` list to look for desired tweet contexts and annotations. 

Simply navigate to the bots directory with ```cd bots```, then run ```python3 tweetdb.py```. Adjust the time buffer in ```time.sleep()``` for optimal time saving with no timeouts. 

### Running Local - Models Setup
1. Clone *llama.cpp* github repo into the project folder, this will be only used to convert *.pth files to *.bin files and to quantize them into 4-bit. To convert and quantize the models follow the instructions found on the *llama.cpp* README (everything working fine as of 04/04/2023).
2. **GPT4All**: download the model from the *gpt4all* github repo referenced in the *Resources* section. Convert the model to ggml format and quantize it, then save the model path in the environment variable ```GPT4ALL_PATH```. 
3. **Llama Models**: download the models from the internet, convert and quantize them following the instructions found in the *llama.cpp* repo, then save the model path in the environment variables ```LLAMA_7B_PATH``` and ```LLAMA_13B_PATH```. 


## **Additional Info**
### Runtimes
Machine: Macbook Pro M1 Pro 8 Cores 16GB.

100 Generations on ~250-token prompt:
- BLOOM (api): 5 min 9 sec
- GPT4All (local, 6 threads): 32 min 26 sec
---

## **Resources**
- **Twitter API Docs**: https://developer.twitter.com/en/docs/twitter-api
- **Twitter context annotations**: https://github.com/twitterdev/twitter-context-annotations/tree/main/files
- **Langchain Docs**: https://python.langchain.com/en/latest/index.html
- **GPT4ALL Repo**: https://github.com/nomic-ai/gpt4all
- **llama.cpp**: https://github.com/ggerganov/llama.cpp
# **msc-thesis-llm-clustering**

## **Aim of this study**
TODO

---

## **TODOs**
TODO

---

## **Instructions**
### Configurations
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

---

## **Resources**
- **Twitter API Docs**: https://developer.twitter.com/en/docs/twitter-api
- **Twitter context annotations**: https://github.com/twitterdev/twitter-context-annotations/tree/main/files
- **Langchain Docs**: https://python.langchain.com/en/latest/index.html
- **GPT4ALL Repo**: https://github.com/nomic-ai/gpt4all
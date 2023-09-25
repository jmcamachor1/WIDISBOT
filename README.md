# WIDISBOT

This is the code for the tools presented in the published paper: [WIDISBOT: Widget to analyse disinformation and content spread by bots](https://research.unl.pt/ws/portalfiles/portal/71487771/LDK2023.pdf). 

## Model

This app makes use of two models. The model used depends on the version of the input Tweet Object.

## Use

In order to use this app it is needed to prepare a suitable Python environment. Follow the next steps to install it using a conda environment.

In the Anaconda Terminal (or Linux and Mac Terminal). Initialize a new environment:
```
conda create --name {env_name}
```

Activate it
```
conda activate {env_name}
```

And install the necessary requirements.
```
conda install --file requirements.txt
```

Start the app via
```
streamlit run Data_loading.py
```

## Tools

- Data extraction 

Connect with Twitter API to extract relevant tweets. In order to use this functionality you need the Bearer Token associated with a working Twiter API app.

This extraction is subject to the limitations of Twitter API. So only data from the last 7 days can be retrieved. With the exception of academic access which allows the retrieval of all Twitter data via the 2.0 API.

- Monitoring

Given a list of tweets, analyze the probability of being produced by a bot and the distribution of bot vs. human authors.

- Forensics

Given a list of usernames computes the likelihood of them being bots.

- Sentiment Analysis

Given a list of tweets gives distribution of the sentiment behind those produced by a bot, or by a human.

- Hashtag Analysis

Visualization of most used hashtags by bot and human accounts given a list of tweets.

- Wordcloud

Visualization of most used words by bot and human accounts given a list of tweets.

- Analysis of spread sources 

Visualization of most used URLs by bot and human accounts given a list of tweets. Connected with third-party Wayback Machine and Media Bias Fact Check to determine the reliability of said URLs.

- Analysis of discourse around hashtags

Allows the use of previous tools centered on the account using a specific hashtag.

## Important notes

- WIDISBOT is intended for academic research purposes only, excluding commercial use.

- WIDISBOT can utilize tweets extracted through both Twitter API v1.1 and Twitter API V2, with various access options available (Basic, Pro, Enterprise, Academic). The application is subject to Twitter's rate limits and policy.

- Any modifications to the functionality of the Twitter API may impact the capabilities of WIDISBOT.







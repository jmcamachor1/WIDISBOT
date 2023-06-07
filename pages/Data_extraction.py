import streamlit as st
import pandas as pd
import tweepy
import datetime
import math
from utils_functions import adapt_user


LANG_DICT = {
 'Spanish': 'es',
 'French': 'fr',
 'English': 'en',
 'Arabic': 'ar',
 'Japanese': 'ja',
 'German': 'de',
 'Italian': 'it',
 'Indonesian': 'id',
 'Portuguese': 'pt',
 'Korean': 'ko',
 'Turkish': 'tr',
 'Russian': 'ru',
 'Dutch': 'nl',
 'Filipino': 'fil',
 'Malay': 'msa',
 'Traditional Chinese': 'zh-tw',
 'Simplified Chinese': 'zh-cn',
 'Hindi': 'hi',
 'Norwegian': 'no',
 'Swedish': 'sv',
 'Finnish': 'fi',
 'Danish': 'da',
 'Polish': 'pl',
 'Hungarian': 'hu',
 'Farsi': 'fa',
 'Hebrew': 'he',
 'Urdu': 'ur',
 'Thai': 'th',
 'English UK': 'en-gb'}

#@st.cache_data
@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')


st.markdown('### Data extraction')


st.markdown('Insert your Twitter App credentials')

api_version = st.selectbox('Select which version of the Twitter API to use.', ("API v2.0", "API v1.1"))
token = st.text_input('Twitter App token')
if token != "":
    if api_version == "API v1.1":
        try:
            auth = tweepy.OAuth2BearerHandler(token)
            api = tweepy.API(auth, wait_on_rate_limit = True)
        except Exception as e:
            st.warning(f'Unexpected error while connecting with the API.\n{e}')
    elif api_version == "API v2.0":
        try:
            api = client = tweepy.Client(
                bearer_token=token,
                wait_on_rate_limit=True
            )

        except Exception as e:
            st.warning(f'Unexpected error while connecting with the API.\n{e}')

    st.markdown('#### Using tweets IDs ')

    st.markdown('__Insert a .txt file with a list of tweet IDs to analyze__')

    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        numbers = uploaded_file.read().decode("utf-8").splitlines()



        tweet_ids = numbers

        if api_version == "API v1.1":    
            A = []
            for t in tweet_ids:
                try:
                # Retrieve tweets
                    tweet_retrieved = api.get_status(t)
                    A.append(tweet_retrieved)
                except tweepy.errors.HTTPException as e:
                    if e.api_messages[0] == 'No status found with that ID.':
                        pass

            retrieved_df = pd.DataFrame([element._json for element in A])
        elif api_version == "API v2.0":
            try:
            # Retrieve tweets
            # TODO: Puede dar error por diferencia en modo de mostrar tweets. Por ver si da.
                info = api.get_tweets(
                    tweet_ids,
                    tweet_fields=[
                        'created_at', 
                        'entities', 
                        'geo', 
                        'lang', 
                        'public_metrics',
                    ],
                    expansions=['author_id'],       
                    user_fields=[
                        "location",
                        "url",
                        "description",
                        "protected",
                        "public_metrics",
                        "created_at",
                        "entities",
                        "profile_image_url",
                        "verified"
                    ])
                
                authorid_to_username = {
                    str(u.id): adapt_user(u.data) for u in info[1]['users']
                } 
                tweets_retrieved = []
                for tweet in info[0]:
                    tweet = tweet.data 
                    
                    created_at = datetime.datetime.strptime(tweet["created_at"], "%Y-%m-%dT%H:%M:%S.%f%z")
                    tweet["created_at"] = created_at.strftime("%a %b %d %H:%M:%S %z %Y")
                    tweet['user'] = authorid_to_username[tweet['author_id']]
                    tweets_retrieved.append(tweet)
            except tweepy.errors.HTTPException as e:
                if e.api_messages[0] == 'No status found with that ID.':
                    pass

            retrieved_df = pd.DataFrame(tweets_retrieved)
            
        st.write('The downloaded dataset contains ', len(retrieved_df), 'tweets.')

        csv_file_retrieved = convert_df(retrieved_df)


        st.download_button(
            label="Download tweets in .csv file from IDs",
            data= csv_file_retrieved,
            file_name= 'data_extracted_IDs.csv',
            mime='text/csv',
            key='download-csv'
        )
    else:
        st.warning('You need to upload the tweets IDs...')
    st.markdown('#### Using kewywords, hashtags or URL ...')

    st.markdown('__Search according to the keywords, hashtag or URL (comma spearated) that you want to investigate, the number of tweets that you want to extract, the language of the tweets and the date from which start considering...__')


    st.markdown("__IMPORTANT:__ Standard access Twitter API only allows retrieval of tweets from the last 7 days. With the exception of Academic access to API v2.0 that allows access to the whole Twitter archive.")

    sw = st.text_input('Some of this words').split(",")

    aw = st.text_input('All of this words').split(",")

    kw = st.text_input('Specific hashtags').split(",")

    u = st.text_input('Specific urls').split(",")

    n_tweets = int(st.number_input('Max tweets',value=100))

    lang_ = st.selectbox('Language of the tweets', LANG_DICT.keys())

    if api_version == "API v1.1":
        until = st.text_input('Until date (not included) (Format: yyyy-mm-dd)', value="")
    elif api_version == "API v2.0":
        since = st.text_input('Since date (Format: yyyy-mm-dd)', value="")
        until = st.text_input('Until date (not included) (Format: yyyy-mm-dd)', value="")


    q = " ".join(["{}".format(w) for w in sw if w != ""] + 
                ["\"{}\"".format(w) for w in aw if w != ""] + 
                ["#{}".format(w) for w in kw if w != ""] + 
                ["url:{}".format(w) for w in u if w != ""])
    print(q)
    if len(q)>500:
        st.warning('This parameters made up a too big query, please take some down.')

    if st.button("Search"):
        if api_version == "API v1.1":   
            tweets = tweepy.Cursor(api.search_tweets,
                                            q, lang=LANG_DICT[lang_],
                                            until=until,
                                            tweet_mode='extended').items(n_tweets)

            list_tweets = [tweet._json for tweet in tweets]

            df = pd.DataFrame(list_tweets)

            st.write('The downloaded dataset contains ', len(df), 'tweets.')

            csv_file = convert_df(df)

            #name_str = st.text_input('Insert the name of the file ...')

            st.download_button(
                label="Download tweets in .csv file from URLs, keywords ...",
                data=csv_file,
                file_name= 'data_extracted_kw.csv',
                mime='text/csv',
                key='download-csv2'
            )
        elif api_version == "API v2.0":


            if since != "":
                dt_min_date = datetime.date.fromisoformat(since)
                min_date_iso = dt_min_date.strftime('%Y-%m-%dT%H:%M:%SZ')  # ISO 8601
            else:
                min_date_iso = None
            if until != "":
                dt_max_date = datetime.date.fromisoformat(until)
                max_date_iso = dt_max_date.strftime('%Y-%m-%dT%H:%M:%SZ')  # ISO 8601
            else:
                max_date_iso = None



            q = q + f" lang:{LANG_DICT[lang_]}"
            list_tweets=[]
            for tweets in tweepy.Paginator(api.search_all_tweets,
                                        query=q, 
                                        start_time=min_date_iso,
                                        end_time=max_date_iso,
                                        max_results=500,
                                        tweet_fields=[
                                            'created_at', 
                                            'entities', 
                                            'geo', 
                                            'lang', 
                                            'public_metrics',
                                        ],    
                                        expansions=['author_id'],       
                                        user_fields=[
                                            "location",
                                            "url",
                                            "description",
                                            "protected",
                                            "public_metrics",
                                            "created_at",
                                            "entities",
                                            "profile_image_url",
                                            "verified"
                                        ],
                                        limit=math.ceil(n_tweets / 500)):
                # Control for empty results    
                if tweets.meta["result_count"] == 0:
                    break

                # The u.id is returned as an int in includes but as str in the tweet
                # object
                authorid_to_username = {
                    str(u.id): adapt_user(u.data) for u in tweets[1]['users']
                } 
            
                for tweet in tweets[0]:
                    tweet = tweet.data 
                    
                    created_at = datetime.datetime.strptime(tweet["created_at"], "%Y-%m-%dT%H:%M:%S.%f%z")
                    tweet["created_at"] = created_at.strftime("%a %b %d %H:%M:%S %z %Y")
                    tweet['user'] = authorid_to_username[tweet['author_id']]
                    list_tweets.append(tweet)

            list_tweets= list_tweets[:n_tweets]

            df = pd.DataFrame(list_tweets)
            
            st.write('The downloaded dataset contains ', len(df), 'tweets.')

            csv_file = convert_df(df)

            name_str = st.text_input('Insert the name of the file ...',value="data_extracted_kw.csv", label_visibility="hidden")

            st.download_button(
                label="Download tweets in .csv file from URLs, keywords ...",
                data=csv_file,
                file_name= name_str,
                mime='text/csv',
                key='download-csv2'
            )
else:
    st.warning("Please provide a valid Twitter API Bearer Token")
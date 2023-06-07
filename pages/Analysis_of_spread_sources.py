import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from prototype_starlight_class import *
import webbrowser
import re

def extract_urls(tweet):
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet)
    return urls



st.markdown("### Source analysis")

st.markdown('#### URL extraction')


st.markdown('__Extract a file with csv from which you want to extract URLs...__')

model_dir = 'botometer_light.pkl' 
uploaded_file = st.file_uploader("Choose a file")
api_version = st.selectbox('Select which version of the Twitter API to use.', ("API v2.0", "API v1.1"))
if api_version=="API v2.0":
    model_dir = 'botometer_light_v2_streamlit.pkl' 
    api_v2=True
else:
    api_v2=False
    


if uploaded_file is not None:

    bot_threshold = st.number_input('Insert threshold to decide if bot or human')
    st.markdown(" __NOTE:__ An account is labeled as human or bot using a threshold. If the probability that an account is a bot is higher/lower than the threshold, it is classified as a bot/human. Therefore, the higher the threshold, the model is more restrictive to assign an account the bot label.")

    psc = proto_starl_class(tweets_dir = uploaded_file,
                        bot_detector_dir = model_dir ,
                        bot_thres = bot_threshold,
                        sentiment_b= True,  
                        cols = [],api_v2=api_v2)



    st.markdown('__Insert the number of urls...__')

    num_urls = int(st.number_input('# of urls.', value=10))


    st.write('#### General')
    psc.reset()
    #full_text_l = psc.df['full_text']
    url_lists = list(psc.df['full_text'].apply(extract_urls))
    plain_url_l = []

    for url_l in url_lists:
        for url in url_l:
            plain_url_l.append(url)

    url_counter = Counter(plain_url_l).most_common(num_urls)
    url_show_df = pd.DataFrame(url_counter,columns = ['url','occurrences'])
    url_str_counter_l = (str(url_counter))
    st.write(url_show_df)
    st.download_button('Download most shrared URL in tweets', url_str_counter_l)

    st.write('#### Humans')
    psc.reset()
    #full_text_l = psc.df['full_text']
    psc.update_df_with_bot_human(label='human')
    url_lists = list(psc.df['full_text'].apply(extract_urls))
    plain_url_l = []
    for url_l in url_lists:
        for url in url_l:
            plain_url_l.append(url)
    url_counter = Counter(plain_url_l).most_common(num_urls)
    url_show_df = pd.DataFrame(url_counter,columns = ['url','occurrences'])
    url_str_counter_l = (str(url_counter))
    st.write(url_show_df)
    st.download_button('Download most shrared URL by humans in tweets', url_str_counter_l)


    st.write('#### Bots')
    psc.reset()
    #full_text_l = psc.df['full_text']
    psc.update_df_with_bot_human(label='bot')
    url_lists = list(psc.df['full_text'].apply(extract_urls))
    plain_url_l = []
    for url_l in url_lists:
        for url in url_l:
            plain_url_l.append(url)
    url_counter = Counter(plain_url_l).most_common(num_urls)
    url_show_df = pd.DataFrame(url_counter,columns = ['url','occurrences'])
    url_str_counter_l = (str(url_counter))
    st.write(url_show_df)
    st.download_button('Download most shrared URL by bots in tweets', url_str_counter_l)

else :
    st.warning("You need to upload the tweets to analyze.")


st.markdown('### Internet archive wayback machine')

st.markdown('__Insert the URL that you want to explore in the internet archive ...__')

g = st.text_input("Insert the URL", key = 6)


url1 = 'https://web.archive.org/web/'+g

if st.button('Open browser',key =9):
    webbrowser.open_new_tab(url1)


st.markdown('### Media Bias Fact Check')


st.markdown('__Insert the name of the media that you want to know its bias ...__')

e = st.text_input("Insert the media name", key = 5)

f1 = e.split(' ')
f2 = '+'.join(f1)
url  = 'https://mediabiasfactcheck.com/?s='+f2

if st.button('Open browser', key= 7):
    webbrowser.open_new_tab(url)




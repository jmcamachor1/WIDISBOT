import streamlit as st
import pandas as pd
import numpy as np
from prototype_starlight_class import *

st.markdown("### Hashtag analysis")

model_dir = 'botometer_light.pkl' 
uploaded_file = st.file_uploader("Choose a file")
api_version = st.selectbox('Select which version of the Twitter API to use.', ("API v2.0", "API v1.1"))
if api_version=="API v2.0":
    model_dir = 'botometer_light_v2_streamlit.pkl' 
    api_v2=True
else:
    api_v2=False

if uploaded_file is not None:
    

    bot_threshold = st.number_input('Insert threshold to decide if the account is bot or human.')
    
    st.markdown("__IMPORTANT:__ An account is labeled as human or bot using a threshold. If the bot score of an account is higher/lower than the threshold, it is classified as a bot/human, respectively. Therefore, the higher the threshold, the more restrictive the model is in assigning the bot label to an account.")

    num_hastags = int(st.number_input('Insert number of hashtags to consider.', value=10))
    psc = proto_starl_class(tweets_dir = uploaded_file,
                        bot_detector_dir = model_dir ,
                        bot_thres = bot_threshold,
                        sentiment_b= False,  
                        cols = [],api_v2=api_v2)
    
    if st.button('General'):
        psc.reset()
        cg = psc.compute_most_frequent_hashtags(num_hastags)
        fig, ax = plt.subplots(figsize=(5, 4))
        labels, values = zip(*cg.items())
        # Create the bar chart
        plt.bar(labels, values)
        # Add labels and title
        plt.xlabel('Hashtags')
        plt.xticks(rotation=90)
        plt.ylabel('Count')
        plt.title('Most frequent hashtags in tweets', size=16)
        st.pyplot(fig)
        cg_str = str(Counter(cg).most_common(num_hastags))
        st.download_button('Download most frequent hashtags in tweets', cg_str)

        
    if st.button('Human'):
        psc.reset()
        psc.update_df_with_bot_human(label='human')
        ch = psc.compute_most_frequent_hashtags(num = num_hastags)
        ch_str = str(Counter(ch).most_common(num_hastags))
        fig, ax = plt.subplots(figsize=(5, 4))
        labels, values = zip(*ch.items())

        # Create the bar chart
        plt.bar(labels, values)

        # Add labels and title
        plt.xlabel('Hashtags')
        plt.xticks(rotation=90)
        plt.ylabel('Count')
        plt.title('Most frequent hashtags in tweets by humans', size=16)
        st.pyplot(fig)
        st.download_button('Download most frequent hashtags in tweets by humans', ch_str)

    if st.button('Bot'):
        psc.reset()
        psc.update_df_with_bot_human(label='bot')
        cb = psc.compute_most_frequent_hashtags(num = num_hastags)
        fig, ax = plt.subplots(figsize=(5, 4))
        labels, values = zip(*cb.items())
        # Create the bar chart
        plt.bar(labels, values)
        cb_str = str(Counter(cb).most_common(num_hastags))
        # Add labels and title
        plt.xlabel('Hashtags')
        plt.xticks(rotation=90)
        plt.ylabel('Count')
        plt.title('Most frequent hashtags in tweets by bots', size=16)
        st.pyplot(fig)
        st.download_button('Download most frequent hashtags in tweets by bots', cb_str)

else:
    st.warning("You need to upload the tweets to analyze.")



import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from prototype_starlight_class import *


st.markdown("### Wordcloud")



model_dir = 'botometer_light.pkl' 
uploaded_file = st.file_uploader("Choose a file")
api_version = st.selectbox('Select which version of the Twitter API to use.', ("API v2.0", "API v1.1"))
if api_version=="API v2.0":
    model_dir = 'botometer_light_v2_streamlit.pkl' 
    api_v2=True
else:
    api_v2=False
    

if uploaded_file is not None:


    bot_threshold = st.number_input('Insert threshold to decide if an account is bot or human.')

    st.markdown("__IMPORTANT:__ An account is labeled as human or bot using a threshold. If the bot score of an account is higher/lower than the threshold, it is classified as a bot/human, respectively. Therefore, the higher the threshold, the more restrictive the model is in assigning the bot label to an account.")

    psc = proto_starl_class(tweets_dir = uploaded_file,
                        bot_detector_dir = model_dir ,
                         bot_thres = bot_threshold,  
                         sentiment_b= False,
                         cols = [],
                         api_v2=api_v2)
    
    if st.button('General'):
        psc.reset()
        fig, ax = plt.subplots(figsize=(8, 6))
        string_ = pd.Series(psc.df['prep_text']).str.cat(sep=' ')
        stopwords = set(STOPWORDS)
        wordcloud = WordCloud(width=1600, stopwords=stopwords,
                                height=800,
                                max_font_size=200,
                                max_words=25,
                                collocations=False,
                                background_color='grey').generate(string_)
        word_dict = str(Counter(wordcloud.process_text(string_)).most_common(25))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(fig)
        st.download_button('Download most frequent words in tweets', word_dict)

    if st.button('Bot'):
        
        psc.reset()
        psc.update_df_with_bot_human(label='bot')
        fig, ax = plt.subplots(figsize=(8, 6))
        string_ = pd.Series(psc.df['prep_text']).str.cat(sep=' ')
        stopwords = set(STOPWORDS)
        wordcloud = WordCloud(width=1600, stopwords=stopwords,
                                height=800,
                                max_font_size=200,
                                max_words=25,
                                collocations=False,
                                background_color='grey').generate(string_)
        word_dict = str(Counter(wordcloud.process_text(string_)).most_common(25))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(fig)
        st.download_button('Download most frequent words in tweets by bots', word_dict)

    
    if st.button('Human'):
        psc.reset()
        psc.update_df_with_bot_human(label='human')
        fig, ax = plt.subplots(figsize=(8, 6))
        string_ = pd.Series(psc.df['prep_text']).str.cat(sep=' ')
        stopwords = set(STOPWORDS)
        wordcloud = WordCloud(width=1600, stopwords=stopwords,
                                height=800,
                                max_font_size=200,
                                max_words=25,
                                collocations=False,
                                background_color='grey').generate(string_)
        word_dict = str(Counter(wordcloud.process_text(string_)).most_common(25))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(fig)
        st.download_button('Download most frequent words in tweets by humans', word_dict)


else:
    st.warning("You need to upload the tweets to analyze.")



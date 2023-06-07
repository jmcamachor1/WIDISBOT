import io
import streamlit as st
import pandas as pd
import numpy as np
from prototype_starlight_class import *



buffer = io.BytesIO()


@st.cache
def convert_df_sentiment(psc_object):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    df = psc_object.reserve_df[['full_text','user_id','screen_name','sentiment_score','sentiment_label','user']]
    df = psc_object.create_bot_columns_in_df(df)
    df = df[['screen_name','full_text', 'bot_score','bot_label', 'sentiment_score', 'sentiment_label']]

    return df





st.markdown("### Sentiment analyzer")

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
    
    psc.df['sentiment_label'] = psc.df['sentiment_score'].apply(psc.compute_sentiment_vader_3_label)
    
    st.write('#### General')
    psc.reset()
    fig = plt.figure(figsize=(20, 4))
    fig, axs = plt.subplots(nrows=1, ncols=2)
    sentiment_l = psc.df['sentiment_score']
    stats_dict = psc.compute_stats(sentiment_l)
    sns.kdeplot(
        sentiment_l,
        bw_adjust=0.5,
        fill=True).set(
        title='Sentiment score')
    plt.xlabel('Sentiment score')
    data = Counter(psc.df['sentiment_label'])
    labels = ['negative', 'neutral','positive']
    values = [data[lab] / sum(data.values()) for lab in labels]
    plt.ylabel('%')
    axs[0].bar(labels, values, color=['red', 'yellow', 'green'])
    axs[0].set_title("Discrete")
    axs[1].yaxis.set_label_position("right")
    axs[1].yaxis.tick_right()
    axs[0].set_xlabel('Label')
    axs[0].set_ylabel('%')
    axs[1].set_xlim(-1, 1)
    axs[1].set_ylabel("Bot score")
    axs[1].set_ylabel("Density")
    axs[1].set_title("Continuous")
    st.pyplot(fig)
    st.write('*__Tweets stats (General)__*')
    st.write("*Tweets with positive/neutral/negative sentiment:*", data['positive'],"/",data['neutral'],'/',data['negative'])
    st.write("*Statistics of the sentiment score distribution:* Maximum:",round(stats_dict['max'],3) ,"Minimum:",round(stats_dict['min'],3), "Mean:", round(stats_dict['mean'],3),"Standard deviation:", round(stats_dict['std'],3),"Median:", round(stats_dict['median'],3),"Q1:", round(stats_dict['q1'],3), "Q3:", round(stats_dict['q3'],3), "IQR:",round(stats_dict['iqr'],3))
    excel_general = convert_df_sentiment(psc)

    


    st.write('#### Humans')
    psc.reset()
    psc.update_df_with_bot_human(label='human')
    fig = plt.figure(figsize=(20, 4))
    fig, axs = plt.subplots(nrows=1, ncols=2)
    sentiment_l = psc.df['sentiment_score']
    stats_dict = psc.compute_stats(sentiment_l)
    sns.kdeplot(
        sentiment_l,
        bw_adjust=0.5,
        fill=True).set(
        title='Sentiment score')
    plt.xlabel('Sentiment score')
    data = Counter(psc.df['sentiment_label'])
    labels = ['negative', 'neutral','positive']
    values = [data[lab] / sum(data.values()) for lab in labels]
    plt.ylabel('%')
    axs[0].bar(labels, values, color=['red','yellow','green'])
    axs[0].set_title("Discrete")
    axs[1].yaxis.set_label_position("right")
    axs[1].yaxis.tick_right()
    axs[0].set_xlabel('Label')
    axs[0].set_ylabel('%')
    axs[1].set_xlim(-1, 1)
    axs[1].set_ylabel("Bot score")
    axs[1].set_ylabel("Density")
    axs[1].set_title("Continuous")
    st.pyplot(fig)
    st.write('*__Tweets stats (Humans)__*')
    st.write("*Tweets with positive/neutral/negative sentiment:*", data['positive'],"/",data['neutral'],'/',data['negative'])
    st.write("*Statistics of the sentiment score distribution:* Maximum:",round(stats_dict['max'],3) ,"Minimum:",round(stats_dict['min'],3), "Mean:", round(stats_dict['mean'],3),"Standard deviation:", round(stats_dict['std'],3),"Median:", round(stats_dict['median'],3),"Q1:", round(stats_dict['q1'],3), "Q3:", round(stats_dict['q3'],3), "IQR:",round(stats_dict['iqr'],3))



    st.write('#### Bots')
    psc.reset()
    psc.update_df_with_bot_human(label='bot')
    fig = plt.figure(figsize=(20, 4))
    fig, axs = plt.subplots(nrows=1, ncols=2)
    sentiment_l = psc.df['sentiment_score']
    stats_dict = psc.compute_stats(sentiment_l)
    sns.kdeplot(
        sentiment_l,
        bw_adjust=0.5,
        fill=True).set(
        title='Sentiment score')
    plt.xlabel('Sentiment score')
    data = Counter(psc.df['sentiment_label'])
    labels = ['negative','neutral' ,'positive']
    values = [data[lab] / sum(data.values()) for lab in labels]
    plt.ylabel('%')
    axs[0].bar(labels, values, color=['red', 'yellow','green'])
    axs[0].set_title("Discrete")
    axs[1].yaxis.set_label_position("right")
    axs[1].yaxis.tick_right()
    axs[0].set_xlabel('Label')
    axs[0].set_ylabel('%')
    axs[1].set_xlim(-1, 1)
    axs[1].set_ylabel("Bot score")
    axs[1].set_ylabel("Density")
    axs[1].set_title("Continuous")
    st.pyplot(fig)
    st.write('*__Tweets stats (Bots)__*')
    st.write("*Tweets with positive/neutral/negative sentiment:*", data['positive'],"/",data['neutral'],'/',data['negative'])
    st.write("*Statistics of the sentiment score distribution:* Maximum:",round(stats_dict['max'],3) ,"Minimum:",round(stats_dict['min'],3), "Mean:", round(stats_dict['mean'],3),"Standard deviation:", round(stats_dict['std'],3),"Median:", round(stats_dict['median'],3),"Q1:", round(stats_dict['q1'],3), "Q3:", round(stats_dict['q3'],3), "IQR:",round(stats_dict['iqr'],3))
    

    st.markdown('### Donwload sentiment results')

    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # Write each dataframe to a different worksheet.
        excel_general.to_excel(writer, sheet_name='Sheet1', index=False)
        # Close the Pandas Excel writer and output the Excel file to the buffer
        writer.save()

    download2 = st.download_button(
        label="Download sentiment results as excel file",
        data=buffer,
        file_name='sentiment_results_'+str(bot_threshold)+'_.xlsx',
        mime='application/vnd.ms-excel'
    )
    
else:
    st.warning("You need to upload the tweets to analyze.")

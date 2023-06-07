import io
import streamlit as st
import pandas as pd
import numpy as np
from prototype_starlight_class import *

st.markdown("### Bot detector")


st.markdown('The probability that an account is a bot is denoted as *bot score*.')


model_dir = 'botometer_light.pkl' 
uploaded_file = st.file_uploader("Choose a file")
api_version = st.selectbox('Select which version of the Twitter API to use.', ("API v2.0", "API v1.1"))
if api_version=="API v2.0":
    model_dir = 'botometer_light_v2_streamlit.pkl' 
    api_v2=True
else:
    #model_dir = 'botometer_light.pkl'
    api_v2=False


@st.cache
def convert_df(psc_object):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    df = psc_object.reserve_df[['full_text','user','screen_name']]
    df = psc_object.create_bot_columns_in_df(df)
    df = df[['screen_name','full_text', 'bot_score','bot_label']]

    return df



buffer = io.BytesIO()




if uploaded_file is not None:
    
    bot_threshold = st.number_input('Insert threshold to decide if an account is bot or human.')

    st.markdown("__IMPORTANT:__ An account is labeled as human or bot using a threshold. If the bot score of an account is higher/lower than the threshold, it is classified as a bot/human, respectively. Therefore, the higher the threshold, the more restrictive the model is in assigning the bot label to an account.")

    psc = proto_starl_class(tweets_dir = uploaded_file,
                        bot_detector_dir = model_dir ,
                         bot_thres = bot_threshold,  
                         sentiment_b= False,
                         cols = [],
                         api_v2=api_v2)
    


    
    fig = plt.figure(figsize=(20, 4))
    fig, axs = plt.subplots(nrows=1, ncols=2)
    bot_count = Counter(psc.bot_pred_df['bot_label'])
    sns.set_style('whitegrid')
    labels = ['human', 'bot']
    values = [bot_count[lab] / sum(bot_count.values())
                      for lab in labels]
    axs[0].bar(labels, values, color=['blue', 'grey'])
    axs[0].set_xlabel('Label')
    axs[0].set_ylabel('%')
    axs[0].set_title('Humans/bots accounts')
    tweets_by_account_type = Counter(psc.df['bot_label'])
    labels = ['human', 'bot']

    values = [tweets_by_account_type[lab] /
                    sum(tweets_by_account_type.values()) for lab in labels]
    axs[1].bar(labels, values, color=['blue', 'grey'])
    axs[1].set_xlabel('Label')
    axs[1].set_ylabel('%')
    axs[1].set_title("Humans'/Bots' tweets")
    st.pyplot(fig)

    fig1, ax1 = plt.subplots(figsize=(5, 4))
    sns.set_style('whitegrid')
    bot_score_list = list(psc.bot_pred_df['bot_score'])
    axs[1].yaxis.set_label_position("right")
    axs[1].yaxis.tick_right()
    sns.kdeplot(
            bot_score_list,
            bw_adjust=0.5,
            fill=True).set(
            title='Bot score probability distribution',
            xlim=0)
    plt.axvline(x = bot_threshold, color = 'b')
    plt.xlabel('Bot score')
    ax1.set_xlim(0, 1)
    stats_dict = psc.compute_stats(bot_score_list)
    st.pyplot(fig1)
    st.markdown("*The vertical blue line represents the inserted threshold.*")
    st.write('Human/Bot accounts:', bot_count['human'], "/", bot_count['bot'])
    st.write('Tweets by humans/bots:', tweets_by_account_type['human'],"/",tweets_by_account_type['bot'])
    st.write("*Below are displayed some relevant statistics of the bot score distribution:* Maximum:",round(stats_dict['max'],3) ,"Minimum:",round(stats_dict['min'],3), "Mean:", round(stats_dict['mean'],3),"Standard deviation:", round(stats_dict['std'],3),"Median:", round(stats_dict['median'],3),"Q1:", round(stats_dict['q1'],3), "Q3:", round(stats_dict['q3'],3), "IQR:",round(stats_dict['iqr'],3))
    excel = convert_df(psc)
    #st.write(psc.bot_pred_df.columns)

    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # Write each dataframe to a different worksheet.
        excel.to_excel(writer, sheet_name='Sheet1', index=False)
        # Close the Pandas Excel writer and output the Excel file to the buffer
        writer.save()

    download2 = st.download_button(
        label="Download bot detection results as excel file",
        data=buffer,
        file_name='bot_detection_results_'+str(bot_threshold)+'_.xlsx',
        mime='application/vnd.ms-excel'
    )




else:
    st.warning("You need to upload the tweets to analyze.")



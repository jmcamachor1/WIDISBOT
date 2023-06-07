import io
import os
import tweepy
import datetime
from prototype_starlight_class import *
from utils_functions import *
import streamlit as st


buffer = io.BytesIO()



st.markdown('### User analysis')

st.markdown('Insert your Twitter App credentials')


model_dir = 'botometer_light.pkl' 
api_version = st.selectbox('Select which version of the Twitter API to use.', ("API v2.0", "API v1.1"))
token = st.text_input('Twitter App token')
if api_version=="API v2.0":
    model_dir = 'botometer_light_v2_streamlit.pkl' 
    api_v2=True
    try:
        api = client = tweepy.Client(
            bearer_token=token,
            wait_on_rate_limit=True
        )
    except Exception as e:
        st.warning(f'Unexpected error while connecting with the API.\n{e}')
else:
    api_v2=False
    try:
        auth = tweepy.OAuth2BearerHandler(token)
        api = tweepy.API(auth, wait_on_rate_limit = True)
    except Exception as e:
        st.warning(f'Unexpected error while connecting with the API.\n{e}')

if token != "":


    bot_detector = pickle.load(open(model_dir, 'rb'))


    bot_threshold = st.number_input('Insert threshold to decide if an account is bot or human.')

    st.markdown('__Insert the user name of the account that you want to analyse__')

    username = st.text_input('Twitter account username.')

    if username != '':

        predict_one_bot_account(username, bot_detector, bot_threshold, api,  display = True, api_v2=api_v2)
        
    else:
        st.warning("You need to type an username to analyze.")


    st.markdown('**Insert a set of users to analyze**')
    uploaded_file = st.file_uploader("Insert a .txt file with user names")


    if uploaded_file is not None:
        usernames_l = uploaded_file.read().decode("utf-8").splitlines()
        
        prediction_df = analyze_set_accounts_from_identifier(usernames_l, bot_detector, api, threshold = bot_threshold, api_v2=api_v2)
        bot_count = Counter(prediction_df['bot_label'])
        fig = plt.figure(figsize=(20, 4))
        fig, axs = plt.subplots(nrows=1, ncols=2)
        sns.set_style('whitegrid')
        labels = ['human', 'bot']
        values = [bot_count[lab] / sum(bot_count.values())
                        for lab in labels]
        axs[0].bar(labels, values, color=['blue', 'grey'])
        axs[0].set_xlabel('Label')
        axs[0].set_ylabel('%')
        axs[0].set_title('Humans/bots accounts')
        tweets_by_account_type = Counter(prediction_df['bot_label'])
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
        bot_score_list = list(prediction_df['bot_score'])
        sns.kdeplot(
                bot_score_list,
                bw_adjust=0.5,
                fill=True).set(
                title='Bot score probability distribution',
                xlim=0)
        plt.axvline(x = bot_threshold, color = 'b')
        plt.xlabel('Bot score')
        ax1.set_xlim(0, 1)
        stats_dict = compute_stats(bot_score_list)
        st.pyplot(fig1)
        st.markdown("*The vertical blue line represents the inserted threshold.*")
        
        st.write('Human/Bot accounts:', bot_count['human'], "/", bot_count['bot'])
        st.write('Tweets by humans/bots:', tweets_by_account_type['human'],"/",tweets_by_account_type['bot'])
        st.write("*Below are displayed some relevant statistics of the bot score distribution:* Maximum:",round(stats_dict['max'],3) ,"Minimum:",round(stats_dict['min'],3), "Mean:", round(stats_dict['mean'],3),"Standard deviation:", round(stats_dict['std'],3),"Median:", round(stats_dict['median'],3),"Q1:", round(stats_dict['q1'],3), "Q3:", round(stats_dict['q3'],3), "IQR:",round(stats_dict['iqr'],3))

        

        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            # Write each dataframe to a different worksheet.
            prediction_df.to_excel(writer, sheet_name='Sheet1', index=False)
            # Close the Pandas Excel writer and output the Excel file to the buffer
            writer.save()

        download2 = st.download_button(
            label="Download bot detection results as excel file",
            data=buffer,
            file_name='bot_detection_results_'+str(bot_threshold)+'_.xlsx',
            mime='application/vnd.ms-excel'
        )
    else:
        st.warning("You need to upload a file.")
else:
    st.warning("Please provide a token")

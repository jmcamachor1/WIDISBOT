import streamlit as st
import pandas as pd
import numpy as np
from prototype_starlight_class import *
import io


@st.cache
def convert_df(psc_object):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    #df = psc_object.reserve_df[['full_text','user','screen_name','user_id']]
    #df = psc_object.create_bot_columns_in_df(df)
    df = psc_object.df[['screen_name','full_text', 'bot_score','bot_label']]

    return df

@st.cache
def convert_df_sentiment(psc_object):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    #df = psc_object.reserve_df[['full_text','user','screen_name','sentiment_score','sentiment_label','user_id']]
    #df = psc_object.create_bot_columns_in_df(df)
    df = psc_object.df[['screen_name','full_text', 'bot_score','bot_label', 'sentiment_score', 'sentiment_label']]

    return df

def extract_urls(tweet):
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet)
    return urls



buffer = io.BytesIO()




st.markdown("### Discourse around hashtags")



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
                        cols = [],
                        api_v2=api_v2)

    psc.df['sentiment_label'] = psc.df['sentiment_score'].apply(psc.compute_sentiment_ncd_2_label)
    hash_list_col = list(psc.df['hashtag_list'])
    hash_counter = Counter([item for sublist in psc.df['hashtag_list'] for item in sublist])
    hashtag_ = st.text_input('Hashtag', 'Insert a hashtag')
    

    if hashtag_ in hash_counter:
        st.write('This hashtag is in the tweets.')
        if st.button('Bot detection'):
            psc.reset()
            psc.update_df_hashtag(hashtag_)
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


        if st.button('Sentiment Analysis'):
            psc.reset()
            psc.update_df_hashtag(hashtag_)
            st.write('#### General')
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
            axs[0].bar(labels, values, color=['red','yellow', 'green'])
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
            excel = convert_df_sentiment(psc)

            st.write('#### Humans')
            psc.reset()
            psc.update_df_with_bot_human(label='human')
            psc.update_df_hashtag(hashtag_)
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
            labels = ['negative','yellow', 'positive']
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
            psc.update_df_hashtag(hashtag_)
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
            labels = ['negative', 'neutral', 'positive']
            values = [data[lab] / sum(data.values()) for lab in labels]
            plt.ylabel('%')
            axs[0].bar(labels, values, color=['red', 'yellow' ,'green'])
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
            st.write("*Tweets with positive/neutral/negative sentiment:*", data['positive'],'/',data['neutral'],"/",data['negative'])
            st.write("*Statistics of the sentiment score distribution:* Maximum:",round(stats_dict['max'],3) ,"Minimum:",round(stats_dict['min'],3), "Mean:", round(stats_dict['mean'],3),"Standard deviation:", round(stats_dict['std'],3),"Median:", round(stats_dict['median'],3),"q1:", round(stats_dict['q1'],3), "q3:", round(stats_dict['q3'],3), "iqr:",round(stats_dict['iqr'],3))

            #st.markdown('### Download sentiment results')
            #excel = convert_df_sentiment(psc)

            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                # Write each dataframe to a different worksheet.
                excel.to_excel(writer, sheet_name='Sheet1', index=False)
                # Close the Pandas Excel writer and output the Excel file to the buffer
                writer.save()

            download2 = st.download_button(
                label="Download sentiment results as excel file ",
                data=buffer,
                file_name='sentiment_results_'+str(bot_threshold)+'_.xlsx',
                mime='application/vnd.ms-excel')

        if st.button('Hashtag analysis'):
            psc.reset()
            psc.update_df_hashtag(hashtag_)
            st.write('#### General')
            cg = psc.compute_most_frequent_hashtags(6)
            del cg[hashtag_]
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
            cg_str = str((Counter(cg).most_common(6)))
            st.download_button('Download most frequent hashtags in tweets - General', cg_str)
            st.write('#### Human')
            psc.reset()
            psc.update_df_with_bot_human(label='human')
            psc.update_df_hashtag(hashtag_)
            ch = psc.compute_most_frequent_hashtags(num = 6)
            del ch[hashtag_]
            fig, ax = plt.subplots(figsize=(5, 4))
            labels, values = zip(*ch.items())
            # Create the bar chart
            plt.bar(labels, values)
            ch_str = str((Counter(ch).most_common(6)))
            # Add labels and title
            plt.xlabel('Hashtags')
            plt.xticks(rotation=90)
            plt.ylabel('Count')
            plt.title('Most frequent hashtags in tweets by humans', size=16)
            st.pyplot(fig)
            st.download_button('Download most frequent hashtags in tweets - Humans', ch_str)
            st.write('#### Bot')
            psc.reset()
            psc.update_df_with_bot_human(label='bot')
            psc.update_df_hashtag(hashtag_)
            cb = psc.compute_most_frequent_hashtags(num = 6)
            del cb[hashtag_]
            fig, ax = plt.subplots(figsize=(5, 4))
            labels, values = zip(*cb.items())
            # Create the bar chart
            plt.bar(labels, values)
            # Add labels and title
            plt.xlabel('Hashtags')
            plt.xticks(rotation=90)
            plt.ylabel('Count')
            plt.title('Most frequent hashtags in tweets by bots', size=16)
            st.pyplot(fig)
            cb_str = str((Counter(cb).most_common(6)))
            st.download_button('Download most frequent hashtags in tweets - Bots', cb_str)



        if st.button('Wordcloud'):
            psc.reset()
            psc.update_df_hashtag(hashtag_)
            st.write('#### General')
            fig, ax = plt.subplots(figsize=(8, 6))
            string_ = pd.Series(psc.df['prep_text']).str.cat(sep=' ')
            stopwords = set(STOPWORDS)
            wordcloud = WordCloud(width=1600, stopwords=stopwords,
                                    height=800,
                                    max_font_size=200,
                                    max_words=25,
                                    collocations=False,
                                    background_color='grey').generate(string_)
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            st.pyplot(fig)
            word_dict_general = str(Counter(wordcloud.process_text(string_)).most_common(25))
            st.download_button('Download most frequent words in tweets', word_dict_general)


            st.write('#### Bot')
            psc.reset()
            psc.update_df_with_bot_human(label='bot')
            psc.update_df_hashtag(hashtag_)
            fig, ax = plt.subplots(figsize=(8, 6))
            string_ = pd.Series(psc.df['prep_text']).str.cat(sep=' ')
            stopwords = set(STOPWORDS)
            wordcloud = WordCloud(width=1600, stopwords=stopwords,
                                    height=800,
                                    max_font_size=200,
                                    max_words=25,
                                    collocations=False,
                                    background_color='grey').generate(string_)
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            st.pyplot(fig)
            word_dict_bot = str(Counter(wordcloud.process_text(string_)).most_common(25))
            st.download_button('Download most frequent words in tweets by bots', word_dict_bot)

    
            st.write('#### Human')
            psc.reset()
            psc.update_df_with_bot_human(label='human')
            psc.update_df_hashtag(hashtag_)
            fig, ax = plt.subplots(figsize=(8, 6))
            string_ = pd.Series(psc.df['prep_text']).str.cat(sep=' ')
            stopwords = set(STOPWORDS)
            wordcloud = WordCloud(width=1600, stopwords=stopwords,
                                    height=800,
                                    max_font_size=200,
                                    max_words=25,
                                    collocations=False,
                                    background_color='grey').generate(string_)
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            st.pyplot(fig)
            word_dict_human = str(Counter(wordcloud.process_text(string_)).most_common(25))
            st.download_button('Download most frequent words in tweets by humans', word_dict_human)

        if st.button('Source analysis'):
            psc.reset()
            st.write('#### General')
            psc.reset()
            psc.update_df_hashtag(hashtag_)
            url_lists = list(psc.df['full_text'].apply(extract_urls))
            plain_url_l = []
            for url_l in url_lists:
                for url in url_l:
                    plain_url_l.append(url)

            url_counter = Counter(plain_url_l).most_common(5)
            url_show_df = pd.DataFrame(url_counter,columns = ['url','occurrences'])
            url_str_counter_l = (str(url_counter))
            st.write(url_show_df)
            st.download_button('Download most shrared URL in tweets', url_str_counter_l)

            st.write('#### Humans')
            psc.reset()
            psc.update_df_with_bot_human(label='human')
            psc.update_df_hashtag(hashtag_)
            url_lists = list(psc.df['full_text'].apply(extract_urls))
            plain_url_l = []
            for url_l in url_lists:
                for url in url_l:
                    plain_url_l.append(url)
            url_counter = Counter(plain_url_l).most_common(5)
            url_show_df = pd.DataFrame(url_counter,columns = ['url','occurrences'])
            url_str_counter_l = (str(url_counter))
            st.write(url_show_df)
            st.download_button('Download most shrared URL by humans in tweets', url_str_counter_l)


            st.write('#### Bots')
            psc.reset()
            psc.update_df_with_bot_human(label='bot')
            psc.update_df_hashtag(hashtag_)
            url_lists = list(psc.df['full_text'].apply(extract_urls))
            plain_url_l = []
            for url_l in url_lists:
                for url in url_l:
                    plain_url_l.append(url)
            url_counter = Counter(plain_url_l).most_common(5)
            url_show_df = pd.DataFrame(url_counter,columns = ['url','occurrences'])
            url_str_counter_l = (str(url_counter))
            st.write(url_show_df)
            st.download_button('Download most shrared URL by bots in tweets', url_str_counter_l)

    else:
        st.write('This hashtag is NOT in the tweets.')

else:
    st.warning("You need to upload the tweets to analyze.")

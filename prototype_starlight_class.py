import os
import re
import pandas as pd
import seaborn as sns
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
from tqdm.notebook import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from botometer_light_functions import *
from sklearn.neighbors import KernelDensity
tqdm.pandas()


def extract_user_screen_name(x):
    """
    Extracts the screen_name from the 'user' field in a tweet
    """
    id = eval(x)['screen_name']
    return '@'+str(id)



class proto_starl_class:

    def __init__(self,
                 tweets_dir,
                 bot_detector_dir,
                 bot_thres,
                 sentiment_b = False,
                 cols=[],
                 api_v2=False):
        """
        Initializes the proto_starl_class with parameters:
        tweets_dir: str, directory of the CSV file containing tweets
        bot_detector_dir: str, directory of the bot detection model
        bot_thres: float, bot detection threshold
        sentiment_b: boolean, if compute tweets sentiment
        cols: list, columns to include in the dataframe
        """
    
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(tweets_dir)

        # select columns if given
        if cols != []:
            df = df[cols]
        else:
            pass
        self.api_v2 = api_v2
        if api_v2:
            df = self.adapt_to_api1(df)
        # Load the bot detection model from file
        self.bot_detector = pickle.load(open(bot_detector_dir, 'rb'))
        self.bot_thres = bot_thres  # Set the bot detection threshold
        # Call the create_bot_score_df method to create a dataframe containing
        # bot scores and labels
        self.bot_pred_df = self.create_bot_score_df(df)
        # Call the create_bot_columns_in_df method to add bot score and bot
        # label columns to the dataframe
        
        df_w_bot_score = self.create_bot_columns_in_df(df)
        # Call the create_hashtag_column method to add a hashtag column to the
        # dataframe
        self.df = self.create_hashtag_column(df_w_bot_score)
        # Call the compute_sentiment_vader_continuos method to add a sentiment
        # column to the dataframe
        if sentiment_b == True:
            #Load the sentiment analysis model from file
            sent_score = []
            self.vader_analyzer = SentimentIntensityAnalyzer()
            for txt in list(self.df['full_text']):
                sent_score.append(self.compute_sentiment_vader_continuos(txt))
            self.df['sentiment_score'] = sent_score
            #self.df['sentiment_score'] =  list(analyzer.predict_proba(self.df['full_text'].to_list())[:,1])
        # Call the wordcloud method to add a prep text column to the dataframe
        self.df['prep_text'] = self.df['full_text'].apply(
            self.cleaning_tweets)
        ### Call the extract screen name method to add a screen_name column
        self.df['screen_name'] = self.df['user'].apply(extract_user_screen_name)
        self.reserve_df = self.df  # set a copy of the dataframe to allow reset
        # Set a copy to copy of the dataframe containing bot scores and labels
        self.reserve_bot_pred_df = self.bot_pred_df
        # Set a label that updated when refering to human/bot
        self.label = None

    @staticmethod
    def compute_stats(dist):
        """
        Computes and returns statistics on a distribution
        """
        min_1 = np.min(dist)
        max_1 = np.max(dist)
        mean = np.mean(dist)
        std = np.std(dist)
        median = np.median(dist)
        q1, q3 = np.quantile(dist, [0.25, 0.75])
        iqr = q3 - q1

        stats_dict = {'min': min_1, 'max': max_1,
                      'mean': mean, 'std': std,
                      'median': median, 'q1': q1,
                      'q3': q3, 'iqr': iqr}

        return stats_dict
    
        
    @staticmethod
    def fix_entities(x):
        """
        Adapts df to api 2
        """

        
        if pd.isna(x):
            x = {}
        else:
            x = eval(x)

        if not "hashtags" in x:
            x["hashtags"] = []
        else:
            for ht in x["hashtags"]:
                ht["text"] = ht["tag"]
        return str(x)
    
    
    def adapt_to_api1(self, df):
        """
        Adapts df to api 2
        """
        df["full_text"] = df["text"]
        df["entities"] = df["entities"].apply(self.fix_entities)
        return df


    def create_bot_score_df(self, df):
        """
        Creates a DataFrame with bot scores using the bot detection model
        """
        prep_tweets_df = extract_user_features_to_baseline(
            df)  # Preprocess the tweets dataframe
        pred_df = evaluate_account(
            prep_tweets_df,
            self.bot_detector,
            self.bot_thres,
            self.api_v2)  # Evaluate the accounts for bot scores using the bot detection model

        return pred_df

    def mapping_id_bot_prob(self):
        """
        Creates a dictionary mapping user IDs to bot scores and labels
        """
        id_list = list(self.bot_pred_df['account_id'])
        bot_score_list = list(self.bot_pred_df['bot_score'])
        bot_label_list = list(self.bot_pred_df['bot_label'])
        map_d = {}

        for i, j in enumerate(id_list):
            map_d[j] = {'label': bot_label_list[i], 'score': bot_score_list[i]}

        return map_d

    @staticmethod
    def extract_user_id(x):
        """
        Extracts the user ID from the 'user' field in a tweet
        """
        id = eval(x)['id']
        return str(id)
    

    def create_bot_columns_in_df(self, df):
        """Add bot label and bot score columns to the DataFrame"""
        map_id = self.mapping_id_bot_prob()
        df['user_id'] = df['user'].apply(self.extract_user_id)
        id_list = list(df['user_id'])
        bot_label_col = [map_id[i]['label'] for i in id_list]
        bot_score_col = [map_id[i]['score'] for i in id_list]
        df['bot_label'] = bot_label_col
        df['bot_score'] = bot_score_col
        return df
    


    def plot_density_bot_score(self, stats=False):
        """Plot the density distribution of the bot score"""
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.set_style('whitegrid')
        bot_score_list = list(self.bot_pred_df['bot_score'])
        sns.kdeplot(
            bot_score_list,
            bw_adjust=0.5,
            fill=True).set(
            title='Bot score density',
            xlim=0)
        plt.xlabel('Bot score')
        if self.label == 'human':
            ax.set_xlim(0, self.bot_thres)
        elif self.label == 'bot':
            ax.set_xlim(self.bot_thres, 1)
        else:
            ax.set_xlim(0, 1)
        if stats:
            stats_dict = self.compute_stats(bot_score_list)
            print(self.compute_stats(bot_score_list))
        else:
            stats_dict = None
        plt.plot()

    def compute_bot_accounts(self, plot=False):
        """
        Compute the count of human and bot accounts from a dataframe and print the results.
        If plot is True, also show a bar chart of the bot vs human account percentages.
        """
        bot_count = Counter(self.bot_pred_df['bot_label'])
        labels = ['human', 'bot']
        print('Human accounts:', bot_count['human'])
        print('Bot accounts:', bot_count['bot'])

        if plot == True:
            values = [bot_count[lab] / sum(bot_count.values())
                      for lab in labels]
            plt.bar(labels, values, color=['blue', 'grey'])
            plt.xlabel('Label')
            plt.ylabel('PCT (%)')

            plt.title('Bots vs Human')

            # Show the chart
            plt.show()

    def compute_tweets_by_account(self, plot=False):
        """
        Compute the count of tweets by human and bot accounts from a dataframe and print the results.
        If plot is True, also show a bar chart of the bot vs human tweet percentages.
        """

        tweets_by_account_type = Counter(self.df['bot_label'])
        labels = ['human', 'bot']
        print('Tweets by humans:', tweets_by_account_type['human'])
        print('Tweets by bots:', tweets_by_account_type['bot'])

        if plot == True:
            values = [tweets_by_account_type[lab] /
                      sum(tweets_by_account_type.values()) for lab in labels]
            plt.bar(labels, values, color=['blue', 'grey'])
            plt.xlabel('Label')
            plt.ylabel('PCT (%)')
            plt.title('Bots tweets  vs Human tweets')
            # Show the chart
            plt.show()

    def create_hashtag_list(self, x):
        """Create a list of hashtags"""
        return [i['text'] for i in eval(x)['hashtags']]

    def create_hashtag_column(self, df):
        """Add a column with a list of hashtags to the DataFrame"""
        df['hashtag_list'] = df['entities'].apply(self.create_hashtag_list)
        return df

    @staticmethod
    def hashtag_boolean(x, h):
        """Check if a hashtag is in a list of hashtags"""
        if h in x:
            b = True
        else:
            b = False
        return b

    def hashtag_boolean_col(self, h):
        """Add a boolean column to the DataFrame indicating if a hashtag is present"""
        self.df['hashtag_boolean'] = self.df['hashtag_list'].apply(
            lambda x: self.hashtag_boolean(x, h))

    @staticmethod
    def linear_hf(x, a=0, b=1):
        """Normalize x between a and b"""
        return (x - a) / (b - a)

    def compute_sentiment_vader_continuos(self, x):
        """Compute the sentiment score using the VADER analyzer"""
        vs = self.vader_analyzer.polarity_scores(x)
        return vs['compound']

    @staticmethod
    def compute_sentiment_vader_2_label(x):
        """Compute the sentiment label with vader thresholds (2 labels)"""

        if x >= 0.5:
            y = 'positive'
        else:
            y = 'negative'
        return y
    
    @staticmethod
    def compute_sentiment_ncd_2_label(x):
        """Compute the sentiment label with ncd (2 labels)"""

        if x >= 0.5:
            y = 'positive'
        else:
            y = 'negative'
        return y


    @staticmethod
    def compute_sentiment_vader_3_label(x):
        """Compute the sentiment label (3 labels)"""

        if x >= 0.05:
            y = 'positive'
        elif x <= -0.05:
            y = 'negative'
        else:
            y = 'neutral'
        return y

    def compute_label_2_bar_chart(self):
        """Compute and display a bar chart that shows the distribution of sentiment
        scores in a dataset, using a 2-label system of 'positive' and 'negative'
        sentiment."""

        data = Counter(
            self.df['sentiment_score'].apply(
                self.compute_sentiment_vader_2_label))
        # Example Counter object with some data
        labels = ['negative', 'positive']
        values = [data[lab] / sum(data.values()) for lab in labels]

        # Extract the labels and values from the Counter object

        print('Tweets with negative sentiment:', data['negative'])
        print('Tweets with positive sentiment:', data['positive'])

        # Create a bar chart using matplotlib
        plt.bar(labels, values, color=['red', 'green'])

        # Add some labels and title to the chart
        plt.xlabel('Sentiment')
        plt.ylabel('PCT (%)')

        plt.title('Sentiment')

        # Show the chart
        plt.show()

    def compute_label_3_bar_chart(self):
        """Compute and display a bar chart that shows the distribution of sentiment
        scores in a dataset, using a 3-label system of 'positive', 'neutral', and
        'negative' sentiment."""

        data = Counter(
            self.df['sentiment_score'].apply(
                self.compute_sentiment_vader_3_label))
        labels = ['negative', 'neutral', 'positive']
        values = [data[lab] / sum(data.values()) for lab in labels]

        print('Tweets with negative sentiment:', data['negative'])
        print('Tweets with neutral sentiment:', data['neutral'])
        print('Tweets with positive sentiment:', data['positive'])

        # Create a bar chart using matplotlib
        plt.bar(labels, values, color=['red', 'yellow', 'green'])

        # Add some labels and title to the chart
        plt.xlabel('Sentiment')
        plt.ylabel('PCT (%)')
        plt.title('Sentiment')

        # Show the chart
        plt.show()

    def compute_most_frequent_hashtags(self, num=20):
        """
        Compute and return a dictionary of the most frequent hashtags in the dataframe.
        """

        all_h = []
        for l in list(self.df['hashtag_list']):
            for h in l:
                all_h.append(h)

        counter_h = Counter(all_h)

        return dict(counter_h.most_common(num))

    def plot_distribution_vader_sentiment_plot(self):
        """Plot a density plot of Vader sentiment scores using seaborn."""

        fig, ax = plt.subplots(figsize=(8, 6))
        #sentiment_l = self.df['sentiment_score'].apply(self.linear_hf)
        sentiment_l = self.df['sentiment_score']
        sns.kdeplot(
            sentiment_l,
            bw_adjust=0.5,
            fill=True).set(
            title='Sentiment score')
        ax.set_xlim(-1, 1)

        plt.plot()

    def plot_most_frequent_hashtags(self, num=20):
        """
        Plot a bar chart of the most frequent hashtags in the dataframe using matplotlib.
        """

        most_freq_d = self.compute_most_frequent_hashtags(num)

        # Extract labels and values from the Counter object
        labels, values = zip(*most_freq_d.items())

        # Create the bar chart
        plt.bar(labels, values)

        # Add labels and title
        plt.xlabel('Label')
        plt.xticks(rotation=90)
        plt.ylabel('Count')
        plt.title('Most frequent hashtags', size=16)

        # Show the chart
        plt.show()

    @staticmethod
    def cleaning_tweets(t):
        """Clean a tweet by removing URLs, mentions, hashtags, emoticons, and returning
        only the text."""

        token = WordPunctTokenizer()
        re_list = [
            '(https?://)?(www\\.)?(\\w+\\.)?(\\w+)(\\.\\w+)(/.+)?',
            '@[A-Za-z0-9_]+',
            '#']
        combined_re = re.compile('|'.join(re_list))
        regex_pattern = re.compile(pattern="["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   "]+", flags=re.UNICODE)

        del_amp = BeautifulSoup(t, 'lxml')
        del_amp_text = del_amp.get_text()
        del_link_mentions = re.sub(combined_re, '', del_amp_text)
        del_emoticons = re.sub(regex_pattern, '', del_link_mentions)
        lower_case = del_emoticons.lower()
        words = token.tokenize(lower_case)
        result_words = [x for x in words if len(x) > 2]
        return (" ".join(result_words)).strip()

    def wordcloud(self):
        """Create wordcloud from preprocessed text"""

        cleaned_tweets = self.df['full_text'].apply(
            self.cleaning_tweets)
        string_ = pd.Series(cleaned_tweets).str.cat(sep=' ')
        stopwords = set(STOPWORDS)
        wordcloud = WordCloud(width=1600, stopwords=stopwords,
                              height=800,
                              max_font_size=200,
                              max_words=50,
                              collocations=False,
                              background_color='grey').generate(string_)
        plt.figure(figsize=(10, 8))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

    def reset(self):
        """Return to the original version of the dataframe"""
        self.df = self.reserve_df
        self.bot_pred_df = self.reserve_bot_pred_df
        self.label = None

    def update_df_hashtag(self, hashtag):
        """Return dataframe updated with hashtag"""
        self.hashtag_boolean_col(hashtag)
        self.df = self.df[self.df['hashtag_boolean'] == True]
        user_id_list = list(self.df['user_id'])
        self.bot_pred_df = self.bot_pred_df[self.bot_pred_df['account_id'].isin(user_id_list)]

    def update_df_with_bot_human(self, label):
        """Return dataframe updated with bot/human label"""
        self.df = self.df[self.df['bot_label'] == label]
        self.bot_pred_df = self.bot_pred_df[self.bot_pred_df['bot_label'] == label]
        self.label = label

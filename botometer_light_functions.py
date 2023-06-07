### load libraries ###

import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import gmean
from sklearn.metrics import f1_score, roc_auc_score, classification_report, accuracy_score, recall_score, confusion_matrix, precision_score, roc_curve, precision_recall_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import warnings
warnings.filterwarnings("ignore")


# Load the dict with the likelihood
# of all bigrams
with open('likelihood_dict.pkl', 'rb') as fp:
    likelihood_dict = pickle.load(fp)


def manual_verified_dummies(df):
    """
    Compute dummies for 'verified' when there is only one label.

    -- Input --
    df: pd.DataFrame
        Dataframe in which to compute the 'Verified' dummies
    -- Output --
    dummies_df: pd.DataFrame
        Dataframe with dummy variables from 'Verified' variable
    """
    len_ = len(df['verified'])

    if set(df['verified']) == {False}:
        dummies_df = pd.DataFrame([1] * len_, columns=['verified_false'], index=df.index)
        dummies_df['verified_true'] = pd.DataFrame([0] * len_, index=df.index)
    elif set(df['verified']) == {True}:
        dummies_df = pd.DataFrame([0] * len_, columns=['verified_false'], index=df.index)
        dummies_df['verified_true'] = pd.DataFrame([1] * len_, index=df.index)


    return dummies_df


def manual_default_profile_dummies(df):
    """
    Compute dummies for 'Default profile' when there is only one label.
    -- Input --
    df: pd.DataFrame
        Dataframe in which to compute the 'Verified' dummies
    -- Output --
    dummies_df: pd.DataFrame
        Dataframe with dummy variables from 'Default profile' variable
    """
    len_ = len(df['default_profile'])
    if set(df['default_profile']) == {False}:
        dummies_df = pd.DataFrame([1] * len_, columns=['dp_false'])
        dummies_df['dp_true'] = pd.DataFrame([0] * len_)
    elif set(df['default_profile']) == {True}:
        dummies_df = pd.DataFrame([0] * len_, columns=['dp_false'])
        dummies_df['dp_true'] = pd.DataFrame([1] * len_)


    return dummies_df


def preprocessing_user_feature_dataframe(df, api_v2):
    """
    Preprocess dataframe for user feature model
    -- Input --
    df: pd.DataFrame
      Dataframe with user features computed from DataFrame
    -- Output --
    df_1: pd.DataFrame
      Preprocessed dataframe for model construction
    """

    df_1 = df.drop('default_profile', axis=1)
    df_1 = df_1.drop(
        labels=[
            'geo_enabled',
            'profile_use_background_image',
            'protected',
            'user_age'],
        axis=1)
    df_1 = df_1.drop('verified', axis=1)

    try:
        one_hot_verified = pd.get_dummies(df['verified'])
        one_hot_verified.columns = ['verified_false', 'verified_true']
    except BaseException:
        one_hot_verified = manual_verified_dummies(df)
    
    if not api_v2:
        try:
            one_hot_default_profile = pd.get_dummies(df['default_profile'])
            one_hot_default_profile.columns = ['dp_false', 'dp_true']
        except BaseException:
            one_hot_default_profile = manual_default_profile_dummies(df)
        df_1 = pd.concat([df_1, one_hot_default_profile, one_hot_verified], axis=1)
    else:
        df_1 = pd.concat([df_1, one_hot_verified], axis=1)

    df_1["followers_friend_ratio"] = df_1["followers_friend_ratio"].fillna(0)
    df_1["followers_friend_ratio"].replace({np.inf: 1000000}, inplace=True)
    return df_1


def evaluate_string_to_variable(x):
    """
    Evaluate string to obtain variable.
    """
    try:
        y = eval(x)
    except BaseException:
        y = x
    return y


def get_number_of_mentions(x):
    """
    Obtain number of mentions.
    """
    try:
        y = len(x['user_mentions'])
    except BaseException:
        y = x
    return y


def get_number_of_hashtags(x):
    """
    Obtain number of hashtags.
    """
    try:
        y = len(x['hashtags'])
    except BaseException:
        y = x
    return y


def get_number_of_urls(x):
    """
    Obtain number of urls.
    """
    try:
        y = len(x['urls'])
    except BaseException:
        y = x
    return y


def tweets_feature_extraction_df(df):
    """
    Extract tweets features.
    --Input --
    df: pd.DataFrame
        Dataframe with tweets.
    --Output--
    df1: pd.DataFrame
        Dataframe with tweet features.
    """
    list_of_columns = ['id', 'retweet_count', 'favorite_count', 'entities']
    df1 = df[list_of_columns]
    df1['entities'] = df1['entities'].apply(evaluate_string_to_variable)
    df1['num_hashtags'] = df1['entities'].apply(get_number_of_hashtags)
    df1['num_urls'] = df1['entities'].apply(get_number_of_urls)
    df1['num_mentions'] = df1['entities'].apply(get_number_of_mentions)
    df1 = df1.drop('entities', axis=1)
    return df1


def change_time_format(x):
    """
    Return string as specific time format.
    """
    try:
        return datetime.strptime(x, '%a %b %d %H:%M:%S %z %Y')
    except BaseException:
        return np.nan


def get_statuses_count(x):
    """
    Obtain the number of tweets.
    """
    try:
        y = x['statuses_count']
    except BaseException:
        y = x
    return y


def get_followers_count(x):
    """
    Obtain the followers of the user.
    """
    try:
        y = x['followers_count']
    except BaseException:
        y = x
    return y


def get_friends_count(x):
    """
    Obtain the friends of the user.
    """
    try:
        y = x['friends_count']
    except BaseException:
        y = x
    return y


def get_favourites_count(x):
    """
    Obtain the favourites given by the account.
    """
    try:
        y = x['favourites_count']
    except BaseException:
        y = x
    return y


def get_listed_count(x):
    """
    Obtain number of lists of the account.
    """
    try:
        y = x['listed_count']
    except BaseException:
        y = x
    return y


def get_default_profile(x):
    """
    Obtain if account has default profile.
    """
    try:
        y = x['default_profile']
    except BaseException:
        y = x
    return y


def get_geo_enabled(x):
    """
    Obtain if account has geo enabled.
    """
    try:
        y = x['geo_enabled']
    except BaseException:
        y = x
    return y


def get_profile_use_background_image(x):
    """
    Obtain if account use profile background image.
    """
    try:
        y = x['profile_use_background_image']
    except BaseException:
        y = x
    return y


def get_verified(x):
    """
    Obtain verified status from the account.
    """
    try:
        y = x['verified']
    except BaseException as e:
        y = x
    return y


def get_protected(x):
    """
    Obtain protected boolean from the account.
    """
    try:
        y = x['protected']
    except BaseException:
        y = x
    return y


def get_account_creation_time(x):
    """
    Obtain user creation time.
    """
    try:
        y = x['created_at']
    except BaseException:
        y = x
    return x


def get_account_id(x):
    """
    Obtain account id of the user.
    """
    try:
        y = x['id']
    except BaseException:
        y = x
    return str(y)


def get_name(x):
    """
    Obtain name of the account.
    """
    try:
        y = x['name']
    except BaseException:
        y = x
    return y


def get_screen_name(x):
    """
    Obtain screen name of the account.
    """
    try:
        y = x['screen_name']
    except BaseException:
        y = x
    return y


def get_len_string(x):
    """
    Obtain lenght of string.
    """
    try:
        y = len(x)
    except BaseException:
        y = x
    return y


def get_number_of_digit_in_string(x):
    """
    Obtain number of digits in string.
    """
    try:
        y = len([c for c in x if c.isdigit()])
    except BaseException:
        y = x
    return y


def get_number_of_digit_in_screen_name(x):
    """
    Obtain numbers of digits in screen name.
    """
    try:
        y = len([c for c in x if c.isdigit()])
    except BaseException:
        y = x
    return y


def get_description(x):
    try:
        y = x['description']
    except BaseException:
        y = x
    return y


def get_account_creation_time(x):
    """
    Obtain account creation time.
    """
    try:
        y = change_time_format(x['created_at'])
    except BaseException:
        y = x
    return y


def user_age_to_hours(x):
    y = x.total_seconds() / 3600
    return y


def get_screen_name_likelihood(screen_name, likelihood_dict=likelihood_dict):
    """
    Compute screen name likelihood.
    -- Input --
        screen_name: str
            Screen name
        likelihood_dict: dict
            Dictionary with the likelihood of all bigrams.
    -- Output--
        y: float
          Screen name likelihood.
    """

    try:
        bigram_list = list(map(''.join, zip(screen_name, screen_name[1:])))
        likelihood_list = []
        for i in bigram_list:
            try:
                likelihood_list.append(likelihood_dict[i])
            except Exception as e:
                likelihood_list.append(0)
        y = gmean(likelihood_list)
    except BaseException:
        y = 0
    return y


def covert_timezone(x):
    """
    Convert timezone
    -- Input --
    x: str
       String with the date
    --Output--
    y: Timezone
      Converted timezone
    """
    try:
        y = x.tz_convert(None)
    except BaseException:
        y = x
    return y


def extract_user_features_to_baseline(df, derived_features=True):
    '''
    Extract user features from tweets dataframe.
    -- Input --
    df: pd.DataFrame
        Dataframe with tweets
    derived_features: boolean
        If True compute derived features,
        if False only extract main features
    -- Output --
    df1: pd.DataFrame
        Dataframe with user features.
    '''
    df1 = df[['user']]
    df1['user'] = df1['user'].apply(evaluate_string_to_variable)
    df1['account_id'] = df1['user'].apply(get_account_id)
    df1 = df1.drop_duplicates(['account_id'])
    df1 = df1.dropna(subset=['account_id'])
    df1['statuses_count'] = df1['user'].apply(get_statuses_count)
    df1['followers_count'] = df1['user'].apply(get_followers_count)
    df1['friends_count'] = df1['user'].apply(get_friends_count)
    df1['favourites_count'] = df1['user'].apply(get_favourites_count)
    df1['listed_count'] = df1['user'].apply(get_listed_count)
    df1['default_profile'] = df1['user'].apply(get_default_profile)
    df1['geo_enabled'] = df1['user'].apply(get_geo_enabled)
    df1['profile_use_background_image'] = df1['user'].apply(
        get_profile_use_background_image)
    df1['verified'] = df1['user'].apply(get_verified)
    df1['protected'] = df1['user'].apply(get_protected)

    if derived_features:
        df1['name'] = df1['user'].apply(get_name)
        df1['length_name'] = df1['name'].apply(get_len_string)
        df1['n_digits_name'] = df1['name'].apply(get_number_of_digit_in_string)
        df1 = df1.drop(['name'], axis=1)
        df1['screen_name'] = df1['user'].apply(get_screen_name)
        df1['length_screen_name'] = df1['screen_name'].apply(get_len_string)
        df1['n_digits_screen_name'] = df1['screen_name'].apply(
            get_number_of_digit_in_string)
        df1['screen_name_likelihood'] = df1['screen_name'].apply(
            get_screen_name_likelihood)
        df1 = df1.drop(['screen_name'], axis=1)
        df1['description'] = df1['user'].apply(get_description)
        df1['len_description'] = df1['user'].apply(get_len_string)
        df1 = df1.drop(['description'], axis=1)
        try:
            df1['account_cretion_time'] = df1['user'].apply(
                get_account_creation_time)
            df1['probe_time'] = df['created_at'].apply(change_time_format)
            df1['user_age_datetime'] = (
                df1['probe_time'] - df1['account_cretion_time'])
        except BaseException:
            df1['account_cretion_time'] = df1['user'].apply(
                get_account_creation_time)
            df1['account_cretion_time'] = df1['account_cretion_time'].apply(
                covert_timezone)
            df1['probe_time'] = df['created_at'].apply(covert_timezone)
            df1['user_age_datetime'] = (
                df1['probe_time'] - df1['account_cretion_time'])
        df1['user_age'] = df1['user_age_datetime'].apply(user_age_to_hours)
        df1 = df1.drop(['account_cretion_time', 'probe_time',
                       'user_age_datetime'], axis=1)
        df1['tweet_freq'] = df1['statuses_count'] / df1['user_age']
        df1['followers_growth_rate'] = df1['followers_count'] / df1['user_age']
        df1['friends_growth_rate'] = df1['friends_count'] / df1['user_age']
        df1['favourites_growth_rate'] = df1['favourites_count'] / df1['user_age']
        df1['listed_growth_rate'] = df1['listed_count'] / df1['user_age']
        df1['followers_friend_ratio'] = df1['followers_count'] / \
            df1['friends_count']

    df1 = df1.drop(['user'], axis=1)

    return df1


def prepare_x_y_train(merged_datasets, datasets_selected):
    """
    Obtain merged X_train y_train from specific datasets
    --- Input --
    merged_datasets: pd.DataFrame
      dataset with all the training data
    datasets_selected: tuple
      string of the datasets considered for training
    -- Output --
    X_train: np.array
      Training data
    y_train: np.array
      Label for training data
    """
    train_data = merged_datasets[merged_datasets['dataset'].isin(
        datasets_selected)]
    X_train = train_data.loc[:, train_data.columns.difference(
        ['label', 'account_id'])]
    y_train = train_data['label']
    return X_train, y_train


def get_F1_threshold(model, X_train, X_val, y_train, y_val):
    """
    Obtain the best threshold that maximizes the classifier's F1 score
    according to X_train/X_val partition.
    -- Input --
    model: sklearn classifier
       Classification model in which is based the bot detector
    X_train: np.array
      Train data.
    X_val: np.array
      Validation data.
    y_train: np.array
      Label for train data.
    y_val: np.array
      Label for validation data.
    -- Output --
    best_tresh: float
      Threshold that maximizes the classifier's F1 score according to
      X_train/X_val partition.
    """

    classifier = model.fit(X_train, y_train)
    try:
        y_val_prob = classifier.predict_proba(X_val)[:, 1]
    except BaseException:
        y_val_prob = classifier.decision_function(X_val)
    fpr, tpr, thresholds = roc_curve(y_val, y_val_prob)
    precision, recall, thresholds = precision_recall_curve(y_val, y_val_prob)
    f1_scores = 2 * recall * precision / (recall + precision)
    best_thresh = thresholds[np.argmax(f1_scores)]
    return best_thresh


def create_df_metrics(classifier, threshold, test_data):
    """
    Assess the classifier on several bot/human datasets.
    -- Input --
    classifier: sklearn classifier
      Classification model in which is based the bot detector
    threshold: float
      Threshold to choose
    test_data: pd.DataFra
    -- Output --
    df_results: pd.DataFrame
      Dataframe with the performance metrics
      of the bot detector
    """

    test_datasets = [
        'botwiki-verified',
        'rtbust',
        'gilani',
        'kaiser',
        'stock',
        'midterm']
    dict_of_results = {}
    for df in test_datasets:
        test_data_1 = test_data[test_data['dataset'] == df]
        X_test = test_data_1.loc[:, test_data.columns.difference(
            ['label', 'account_id', 'dataset'])]
        y_true = test_data_1['label']
        try:
            y_scores = classifier.predict_proba(X_test)[:, 1]
        except BaseException:
            y_scores = classifier.decision_function(X_test)
        y_pred = (y_scores >= threshold).astype(bool)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        auc_score = roc_auc_score(y_true, y_scores)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp)
        specifity = (specificity)
        dict_of_results[df] = [
            auc_score,
            f1,
            accuracy,
            recall,
            precision,
            specifity]
    df_results = pd.DataFrame(dict_of_results)
    df_results['metrics'] = [
        'auc_score',
        'f1_score',
        'accuracy',
        'recall',
        'precision',
        'specifity']
    return df_results


def model_evaluation_pipeline(
        model,
        train_data,
        datasets_selected,
        test_data,
        feature_ext,
        name_model=''):
    """
    Obtain performance of bot detector given the classifier and specific datasets.
    -- Input --
    model: sklearn classifier
      Classifier underlying the bot detector
    train_data: pd.DataFrame
      Preprocessed dataframe with train data
    datasets_selected: tuple
      Tuple with the names of the dataset
    test_data: pd.DataFrame
      Preprocessed dataframe with test data
    feature_ext: str
      string indicating the features used
    name_model: str
      string indicating the model name

    -- Output --
    df: pd.DataFrame
    Dataframe with performance metrics.
    """
    X, y = prepare_x_y_train(train_data, datasets_selected)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=X['dataset'])
    X_train = X_train.loc[:, X_train.columns.difference(
        ['label', 'account_id', 'dataset'])]
    X_val = X_val.loc[:, X_val.columns.difference(
        ['label', 'account_id', 'dataset'])]
    X = X.loc[:, X.columns.difference(['label', 'account_id', 'dataset'])]
    threshold = get_F1_threshold(model, X_train, X_val, y_train, y_val)
    classifier = model.fit(X, y)
    df = create_df_metrics(classifier, threshold, test_data)
    df['datasets_tr'] = [str(datasets_selected)] * len(df)
    if name_model == '':
        df['model'] = [str(model.__class__.__name__)] * len(df)
    else:
        df['model'] = [name_model] * len(df)
    df['threshold'] = ['F1'] * len(df)
    df['dat_feature'] = feature_ext
    return df


def threshold_twitter_bot_human(x, threshold=0.5153893575042254):
    """
    Decide if an account is human or bot regarding a threshold.
    -- Input --
    x: float
      Probability of an account being a bot
    threshold: float
      Threshold to decide if an account is a bot or not
    -- Output --
    label: str
      Label  indicating if an account is a bot or human
    """

    if float(x) > threshold:
        label = 'bot'
    else:
        label = 'human'
    return label


def extract_tweets_with_specific_str(api, str_, num_t=100):
    """
    Produce dataframe with tweets with specific tweets
    -- Input --
    api: tweepy API
      API initialised from tweepy library
    str_ : str
      String to query from Twitter API
    num_t: int
      Number of tweets to extract

    -- Output--
    df: pd.DataFrame
      Dataframe with tweets extracted
    """
    results = api.search_tweets(q=str_, count=num_t)
    json_data = [r._json for r in results]
    df = pd.DataFrame(json_data)
    return df


def evaluate_account(dataset_to_pred, classifier, thres=0.5153893575042254, api_v2=False):
    """
    Predict if accounts are bots or humans.
    -- Input --
    dataset_to_pred: pd.DataFrame
      Dataset with the
    classifier: sklearn object
      Trained classifier to predict if an account is human or bot.
    -- Output --
    y1: pd.DataFrame
      Dataframe with the account id, bot score and bot label.
    """

    col_to_pred = [
        'dp_false',
        'dp_true',
        'favourites_count',
        'favourites_growth_rate',
        'followers_count',
        'followers_friend_ratio',
        'followers_growth_rate',
        'friends_count',
        'friends_growth_rate',
        'len_description',
        'length_name',
        'length_screen_name',
        'listed_count',
        'listed_growth_rate',
        'n_digits_name',
        'n_digits_screen_name',
        'screen_name_likelihood',
        'statuses_count',
        'tweet_freq',
        'verified_false',
        'verified_true']
    
    if api_v2:
        col_to_pred = [
            'followers_count',
            'followers_friend_ratio',
            'followers_growth_rate',
            'friends_count',
            'friends_growth_rate',
            'len_description',
            'length_name',
            'length_screen_name',
            'listed_count',
            'listed_growth_rate',
            'n_digits_name',
            'n_digits_screen_name',
            'screen_name_likelihood',
            'statuses_count',
            'tweet_freq',
            'verified_false',
            'verified_true']
        
    x2 = preprocessing_user_feature_dataframe(
        dataset_to_pred, api_v2)

    x2 = x2.dropna().reset_index(drop=True)
    y1 = x2
    x2 = x2[col_to_pred]
    x3 = classifier.predict_proba(x2)[:, [1]]

    y1['bot_score'] = list(x3.ravel())
    y1['bot_label'] = y1['bot_score'].apply(
        lambda x: threshold_twitter_bot_human(
            x, threshold=thres))

    return y1[['account_id', 'bot_score', 'bot_label']]

import os
import tweepy
import datetime 
import streamlit as st
from prototype_starlight_class import *
import pandas as pd



def adapt_user(user_data):
    new_user = {}
    for key, value in {
        "id": "id",
        "id_str": "id",
        "screen_name": "username",
        "location": "location",
        "url": "url",
        "description": "description",
        "protected": "protected",
        "entities": "entities",
        "verified": "verified",
        "profile_image_url_https": "profile_image_url",
        "name": "name"
    }.items():
        if value in user_data:
            new_user[key] = user_data[value]
        else:
            new_user[key] = None
    if "public_metrics" in user_data:
        new_user["followers_count"] = user_data["public_metrics"]["followers_count"]
        new_user["friends_count"] = user_data["public_metrics"]["following_count"]
        new_user["listed_count"] = user_data["public_metrics"]["listed_count"]
        new_user["favourites_count"] = 0
        new_user["statuses_count"] = user_data["public_metrics"]["tweet_count"]
    else:
        new_user["followers_count"] = None
        new_user["friends_count"] = None
        new_user["listed_count"] = None
        new_user["favourites_count"] = None
        new_user["statuses_count"] = None
    
    if "created_at" in user_data:
        created_at = datetime.strptime(user_data["created_at"], "%Y-%m-%dT%H:%M:%S.%f%z")
        new_user["created_at"] = created_at.strftime("%a %b %d %H:%M:%S %z %Y")
    else:
        new_user["created_at"] = None
    return new_user


def extract_user_info_v2(identifier, api):
    if identifier.isnumeric() == False:
        user=api.get_user(username=identifier).data
        identifier = user["id"]
    
    tweets = api.get_users_tweets(
        id = identifier,
        max_results = 5,
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
        exclude="retweets")
    
    info_d = adapt_user(tweets[1]["users"][0].data)
    tw_list = []
    for tweet in tweets[0]:
        tweet = tweet.data
        
        created_at = datetime.strptime(tweet["created_at"], "%Y-%m-%dT%H:%M:%S.%f%z")
        tweet["created_at"] = created_at.strftime("%a %b %d %H:%M:%S %z %Y")
        tweet['user'] = info_d
        tw_list.append(tweet)
        break
    pred_ft = extract_user_features_to_baseline(pd.DataFrame(tw_list))
    
    return info_d, pred_ft



def extract_user_info(identifier, api):
    try:
        if identifier.isnumeric() == True:
            info = api.user_timeline(user_id = identifier,count = 1)[0]._json
        else:
            info = api.user_timeline(screen_name = identifier,count = 1)[0]._json
    except Exception as e:
        print(e.api_messages[0])
        raise e
    
    info_d = dict(pd.DataFrame([info])['user'][0])
    pred_ft = extract_user_features_to_baseline(pd.DataFrame([info]))

    return info_d, pred_ft


def obtain_stats_user_name(info_d, pred_ft, res_df, bot_detector_model, threshold, api_v2=False):
    st.write('**Name:**',info_d['name'], ', **Screen name:**','@' + info_d['screen_name'])
    #res_df = evaluate_account(pred_ft, bot_detector_model, thres = threshold)
    bot_score = res_df['bot_score'].iloc[0]
    bot_label = res_df['bot_label'].iloc[0]
    st.write('**Bot score:**', bot_score, ', **Bot label:**', bot_label)
    
    st.write('**-----------------------Information about the account ------------------------**')
    st.write('**Created at:**',info_d['created_at'])
    st.write('**Description:**',info_d['description'])
    st.write('**Statuses count:**', info_d['statuses_count'], ', **Followers count:**',info_d['followers_count'], ', **Friends count:**',info_d['friends_count'])
    if api_v2:
        st.write('**Listed count:**',info_d['listed_count'], '**Protected:**',info_d['protected'], ',**Verified:**', info_d['verified'])
        st.write('**Screen name likelihood:**', round(pred_ft['screen_name_likelihood'][0],3), ', **Tweet frequency:**', round(pred_ft['tweet_freq'][0],3))
        st.write('**Followers growth rate:**', round(pred_ft['followers_growth_rate'][0],3))
        st.write('**Listed growth rate**', round(pred_ft['listed_growth_rate'][0],3), '**Followers_friend_ratio**', round(pred_ft['followers_friend_ratio'][0],3))
    else:
        st.write('**Listed count:**',info_d['listed_count'], ', **Favourites:**', info_d['favourites_count'], '**Protected:**',info_d['protected'], ',**Verified:**', info_d['verified'])
        st.write('**Screen name likelihood:**', round(pred_ft['screen_name_likelihood'][0],3), ', **Tweet frequency:**', round(pred_ft['tweet_freq'][0],3))
        st.write('**Favourites growth rate:**', round(pred_ft['favourites_growth_rate'][0],3), ', **Followers growth rate:**', round(pred_ft['followers_growth_rate'][0],3))
        st.write('**Listed growth rate**', round(pred_ft['listed_growth_rate'][0],3), ', **Followers_friend_ratio**', round(pred_ft['followers_friend_ratio'][0],3))
        
    

def predict_one_bot_account(identifier, bot_detector_model, threshold, api ,display = True, api_v2=False):
    
    try:
        if api_v2 == True:
            info_df, pred_ft =  extract_user_info_v2(identifier, api)
        else:
            info_df, pred_ft =  extract_user_info(identifier, api)
        res_df = evaluate_account(pred_ft, bot_detector_model, thres = threshold, api_v2=api_v2)
        if display == True:
            obtain_stats_user_name(info_df, pred_ft, res_df, bot_detector_model, threshold, api_v2=api_v2)
        return res_df
    except Exception as e:
        print(e)
        return


def analyze_set_accounts_from_identifier(username_l, bot_detector, api,threshold, api_v2=False):
    
    pred_obj_l = []
    user_name_l = []
    for a in username_l:
        try:
            if api_v2 == True:
                info_df, pred_ft =  extract_user_info_v2(a, api)
            else:
                info_df, pred_ft =  extract_user_info(a, api)
            user_name_l.append(info_df['screen_name'])
            account_result= predict_one_bot_account(a, bot_detector, threshold,api ,display = False, api_v2=api_v2)
            account_result['username'] = [a]
            pred_obj_l.append(account_result)
        except Exception as e:
            st.write(e)
            if 'tweepy.errors' in str(type(e)):
                pass
            
    res_df = pd.concat(pred_obj_l)
    res_df = res_df[['username','bot_score','bot_label']]
    return res_df.reset_index(drop = True)


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

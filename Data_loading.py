import streamlit as st
import pandas as pd
import tweepy


st.markdown("### Data Loading:")

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')



model_dir = 'botometer_light.pkl' 
uploaded_file = st.file_uploader("Choose a file")
#api_version = st.selectbox('Select which version of the Twitter API to use.', ("API v2.0", "API v1.1"))
#if api_version=="API v2.0":
    #model_dir = 'botometer_light_v2_streamlit.pkl' 
    #api_v2=True
#else:
    #api_v2=False


if uploaded_file is None:
    st.warning("You need to upload the tweets to analyze.")

else:

    df = pd.read_csv(uploaded_file)
    if 'full_text' in df:
        st.write('The uploaded file contains', len(df), 'tweets.')
        st.write(df[['full_text','user','entities']].head(5))
    elif 'text' in df:
        st.write('The uploaded file contains', len(df), 'tweets.')
        st.write(df[['text','user','entities']].head(5))
    

st.markdown("__IMPORTANT:__ Due to the nature of the multiple Twitter APIs, it is important to know which API do the tweets used.")
st.markdown("Depending on the origin of the tweets a diferent model will be used to determine the probability of the account being a bot.")
st.markdown("Generally the model used with the info available from API v1.1 performs better.")






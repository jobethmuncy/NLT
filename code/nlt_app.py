import streamlit as st
import pandas as pd 
from gensim.models.doc2vec import Doc2Vec

st.title("Article Search")
st.header("NLT Project")
st.subheader("By: Cynthia, Clay, Jobeth")

# Read in data
df = pd.read_csv('NLT_data/merged_data.csv')

# Load in model
model = Doc2Vec.load("doc2vec.model")

# Function for searching most similar articles by similar keywords
@st.cache
def most_sim_docs(word, n_articles=2, topn=3):
    dfs = []
    sims = []
    
    # Input word with topn of its most similar keywords
    keywords = [word] + [w[0] for w in model.wv.most_similar_cosmul(word, topn=topn)]
    
    # Loop through each keyword
    for term in keywords:
        new_vec = model.infer_vector([term]) # random, different everytime 
        tag_list = model.docvecs.most_similar([new_vec])[0:n_articles]
        
        
        tags = [] 
        for num in tag_list:
            tags.append(num[0])
            sims.append(num[1])
        
        for t in tags:
            dfs.append(df.iloc[t, :])
        
    new_df = pd.DataFrame(dfs)
    new_df['similarity_percentage'] = sims
    
    return new_df

# User input
user_input = st.text_input("Keyword:")
sort_by = st.sidebar.selectbox("Order articles by:",
                              ['None', 'Year Published', 'Alphabetically'])

# Display results as link to article and article title
if user_input != None:
    articles  = most_sim_docs(word = user_input)

# Display in certain order
order = sort_by
if order == 'Year Published':
    articles = articles.sort_values(by='publish_time', ascending=False)
    
for i in range(articles.shape[0]):
    st.markdown(f"[{articles.iloc[i]['title']}]({articles.iloc[i]['url']})")
    st.markdown("[insert topics here]")
    st.markdown(f"{articles.iloc[i]['text_body'][:300]}...")


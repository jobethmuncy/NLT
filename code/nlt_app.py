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

@st.cache
# Function for searching most similar articles by similar keywords
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

# Sidebar search options
sort_by = st.sidebar.selectbox("Order articles by:",
                              ['None', 'Most Recent'])

# Link to LDA visualization
st.sidebar.markdown("[Topic Visualization](http://localhost:8888/view/projects/client_project/lda.html)")

# Display results as link to article and article title
if user_input == '' or user_input not in model.wv.vocab.keys():
    "Please input a keyword."
else:
    articles = most_sim_docs(word = user_input)

# Display in certain order
order = sort_by

if order == 'None' and user_input in model.wv.vocab.keys():
    for i in range(articles.shape[0]):
        st.markdown(f"[{articles.iloc[i]['title']}]({articles.iloc[i]['url']})")
        st.markdown("[insert topics here]")
        st.markdown(f"{articles.iloc[i]['text_body'][:300]}...")

elif order == 'Most Recent' and user_input in model.wv.vocab.keys():
    articles = articles.sort_values(by='publish_time', ascending=False)
    
    for i in range(articles.shape[0]):
        year_pub = articles.iloc[i]['publish_time'][:4]
        st.markdown(f"[{articles.iloc[i]['title']} ({year_pub})]({articles.iloc[i]['url']})")
        st.markdown("[insert topics here]")
        st.markdown(f"{articles.iloc[i]['text_body'][:300]}...")

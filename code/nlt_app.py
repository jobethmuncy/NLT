import streamlit as st
import pandas as pd 
from gensim.models.doc2vec import Doc2Vec

st.title("Article Search")
st.header("NLT Project")
st.subheader("By: Cynthia Chiang, Clay Carson, Jobeth Muncy")

# Read in data
df = pd.read_csv('articles_with_topics.csv')

# Load in model
model = Doc2Vec.load("doc2vec_updated.model", mmap='r')

# Function for searching most similar articles by similar keywords
def most_sim_docs(word, n_articles=5, topn=5):
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
    
    return new_df.drop_duplicates(subset='title')

# User input
user_input = st.text_input("Keyword:")

# SIDE BAR OPTIONS
# Link to LDA visualization
st.sidebar.markdown("[Topic Visualization](http://localhost:8888/view/projects/client_project/lda.html)")

# Search options
sort_by = st.sidebar.selectbox("Order articles by:",
                              ['Most Relevant', 'Most Recent'])

# Search by topic
topic = st.sidebar.selectbox("Search by topic:",
                             ['', 'German', 'Cellular Expression', 'Heart and Lungs', 'Cell Studies', 
                              'At Risk Groups', 'Household Pets', 'Pre-existing Conditions', 
                              'Global Health', 'Hospital Studies', 'Respiratory Conditions', 
                              'Severe Outbreaks','Nervous System', 'Research Studies', 
                              'Symptoms and Treatment', 'Intestinal Reactions', 'Coronaviruses', 'French', 
                              'Vaccines', 'Proteins', 'Tissues and Lesions', 'Italian', 'Detection',
                              'Spanish', 'Farm Animals', 'Host Infection'])

# Display results as link to article and article title
if user_input == '':
    "Please input a keyword."
elif user_input not in model.wv.vocab.keys():
    "Keyword not found. Please enter a different word."
else:
    articles = most_sim_docs(word = user_input)

# Display in certain order
order = sort_by

if order == 'Most Relevant' and user_input in model.wv.vocab.keys() and topic == '':
    for i in range(articles.shape[0]):
        st.markdown(f"[{articles.iloc[i]['title']}]({articles.iloc[i]['url']})")
        st.markdown(f"**Related topics**: {articles.iloc[i]['Keywords']}")
        st.markdown(f"{articles.iloc[i]['text_body'][:300]}...")

elif order == 'Most Recent' and user_input in model.wv.vocab.keys():
    articles = articles.sort_values(by='publish_time', ascending=False)
    
    for i in range(articles.shape[0]):
        year_pub = articles.iloc[i]['publish_time'][:4]
        st.markdown(f"[{articles.iloc[i]['title']} ({year_pub})]({articles.iloc[i]['url']})")
        st.markdown(f"**Related topics**: {articles.iloc[i]['Keywords']}")
        st.markdown(f"{articles.iloc[i]['text_body'][:300]}...")

# Display articles of a certain topic
topic_selected = topic
topic_list = ['German', 'Cellular Expression', 'Heart and Lungs', 'Cell Studies', 'At Risk Groups',
              'Household Pets', 'Pre-existing Conditions', 'Global Health', 'Hospital Studies', 
              'Respiratory Conditions','Severe Outbreaks', 'Nervous System', 'Research Studies', 
              'Symptoms and Treatment', 'Intestinal Reactions', 'Coronaviruses', 'French', 'Vaccines',
              'Proteins', 'Tissues and Lesions', 'Italian', 'Detection', 'Spanish', 'Farm Animals',
              'Host Infection']

topic_dict = {topic: float(i) for i, topic in enumerate(topic_list)}

# Allows you to get all the articles on a certain topic only if is no keyword input
if topic_selected in topic_list and user_input == '':
    article_by_topic = df.loc[df['Dominant_Topic'] == topic_dict[topic_selected], :]
    
    for i in range(article_by_topic.shape[0]):
        st.markdown(f"[{article_by_topic.iloc[i]['title']}]({article_by_topic.iloc[i]['url']})")
        st.markdown(f"{article_by_topic.iloc[i]['text_body'][:300]}...")

# Filter articles search on keyword by topic
if user_input in model.wv.vocab.keys() and topic_selected in topic_list:
    articles_filtered_by_top = articles.loc[articles['Dominant_Topic'] == topic_dict[topic_selected], :]
    
    for i in range(articles_filtered_by_top.shape[0]):
        st.markdown(f"[{articles_filtered_by_top.iloc[i]['title']}]({articles_filtered_by_top.iloc[i]['url']})")
       # st.markdown(f"**Related topics**: {articles_filtered_by_top.iloc[i]['Keywords']}")
        st.markdown(f"{articles_filtered_by_top.iloc[i]['text_body'][:300]}...")
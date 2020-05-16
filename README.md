# Problem Statement
2020 brought the global economy to a halt with the outbreak of the COVID-19 virus. In response to the pandemic, the Semantic Scholar team at the Allen Institute for AI created CORD-19, the COVID Open Research Dataset, in partnership with Georgetown University, Microsoft Research, Amazon Web Services, the Chan Zuckerberg Initiative, the National Institutes of Health, and the White House. CORD-19 is a database of over 63,000 scientific journal articles relating to coronaviruses and similar outbreaks, dating back to 1957.  

After consolidating the research articles into a downloadable format, the Allen Institute requested help from the machine learning community to develop tools in order to analyze the articles and aid researchers in finding a vaccine. In this project, we explore effective means for topic classification and organization of the literature by using Natural Language Processing, and then create a search engine for efficiently searching through the articles by keyword or topic.

# Executive Summary 
In this project, we explore over 63,000 scientific journal articles hosted on the Allen Institute [website](https://www.semanticscholar.org/cord19/get-started). The data came in two sections: the JSON form, organized by each respective journal, containing the journal name, url, paper ID number, title, authors, and full body text for each article, and a dataframe containing the metadata for all articles. From there, we created a singular  dataframe containing all of the articles and their body text, merged with the metadata dataframe. We created a function to extract the "discussion" section from each respective body text, and if the body text had no discernible discussion section, we imputed those values with the abstract from that particular article.  In order to analyze the data using the Gensim library, the text required substantial cleaning. We wrote a function to process each word from each document, whereby it was stripped of any lasting html artifacts, punctuation, whitespace, and commonly use english stopwords such as "and", "the", "or", etc.   

Once the text was cleaned, we fit a Doc2Vec model on the data. This model assigned vectors to each word and document in order to compare embeddings and relationships for each word and/or document. By mapping the documents in vector-space, the model is able to better learn contextual and semantic relationships between the documents and their contained words. The Doc2Vec model, first preposed [here](https://arxiv.org/pdf/1405.4053v2.pdf) is a continuation of the popular Word2Vec model, first preposed by Google engineers [here](https://arxiv.org/pdf/1301.3781.pdf) and [here](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf). While Doc2Vec makes comparison between document and word vectors easy and efficient, one of the downsides of the model is its lack of interpretability in terms of its transformation of the words themselves.

For a more interpretable analysis of the documents, we employed a Latent Dirichlet Allocation. This particular model assumes that each document is made up of a mixture of topics, each gleaned from a weighted mixture of uniquely occuring words. Each document is then returned with an accompanying list of possible topics. Besides assigning topics which are then comparable between documents, however, LDA does not provide means for quantitatively comparing the literature.

Our method for parsing the two models was to create a search engine the returned results based on a keyword. The keyword is first passed to the Doc2Vc model, where the vector of the keyword is inferred from the trained model. That inferred vector is compared to the collection of document vectors, and the most similar documents are returned based on their own vectors. Additionally, the keyword's word vector is compared to similar word vectors, and those word vectors are compared to document vectors for which more documents are returned, effectively multiplying the relevance of the search query. LDA is then used to optionally reduce the returned documents if the user specifies a topic by which to sort the documents. The search engine itself was hosted on a local machine by using the Streamlit library, but in full production would be hosted on the web to optimize its speed and make it available to those researchers who need its utility. 

# Project Directory

<h4>code</h4>
    <ul>
        <li>app</li>
        <li>01_reformatting_text_data.ipynb</li>
        <li>02_filtering_dataset.ipynb</li>
        <li>03_merge_filtered_data.ipynb</li>
        <li>03_merge_metadata.ipynb</li>
        <li>04_word2vec_filtered_data.ipynb</li>
        <li>04_word2vec_metadata.ipynb</li>
        <li>05_doc2vec_recommender.ipynb</li>
        <li>06_doc2vec_recommender_updated.ipynb</li>
        <li>07_LDA_mallet_models.ipynb</li>
        <li>08_add_topics.ipynb</li>
    </ul>
<h4>data</h4>
    <ul>
        <li>coronavirus_articles.csv</li>
        <li>filtered_coronavirus_articles.csv</li>
    </ul>

<h4>images</h4>
    <ul>
        <li>lda.html</li>
        <li>use_lda.html</li>
    </ul>
    NLT-Cynthia, Clay, Jobeth.pdf<br>
    README.md<br>
    links_to_data_and_models.txt

# Conclusions

After utilizing Word2Vec, Doc2Vec, and LDA to process articles from 1957 to present day, the Streamlit application allowed the user to search for articles related to key word, topic, or both. This could help researchers expedite the process of finding papers relevant to their work in the ongoing study of coronaviruses. 
The Word2Vec model found words that were closely related to the searched term and then used Doc2Vec to find articles related to these words. Direct links to the articles were included in the application for immediate access to the data. The LDA process grouped articles together by topic similarity. The user could add a topic to their keyword search to be even more specific. 
If a key word was searched for but was not in the dictionary of learned words, no articles would be shown. A user could instead search by topic or try a different word. Overall, the application was successful in taking key words and topics and finding the most relevant articles from a collection of over 63,000 scientific papers. 

# Recommendations

For further research, the vocabulary should be increased. The models currently have a vocabulary of 769,895 unique words. By expanding the vocabulary, the Streamlit application would be able find articles for a keyword even if that word was not found in any of the documents. Word2Vec would find words similar to the keyword and then allow Doc2Vec to pull articles that were similar to those words.

Additional steps could be taken in the LDA model as well. This model clustered 4 topics separately from the the larger grouping. It was discovered that these are foreign languages. They were grouped together by Spanish, French, German, and Italian. The most prevalent words in each of the clusters were the stopwords from each language. In the data cleaning process, foreign stopwords should be removed just like the English stopwords were. After this, the LDA model should be ran to see how the topic assignment changes. The articles may need to be translated so they are assigned to a relevant topic. 

Currently, the Streamlit application has reruns the model each time it looks for a new word. This takes about 12 seconds each time. If this application was launched, the model should be hosted separately to increase the speed of the returned articles. 
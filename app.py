import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd


st.set_page_config(
    page_title="Amazon Semantic Search",
    page_icon=":bookmark_tabs:",
    layout="wide",
    initial_sidebar_state="expanded", 
)

@st.cache_data
def load_data():
    data = pd.read_csv('amazonFood.csv')
    return data

df = load_data()
review_data = df[['ProductId', 'productName', 'Score', 'Summary','Text']]

def menu():
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sentence_transformers import SentenceTransformer, util
    from sklearn.neighbors import NearestNeighbors  
    import tensorflow_hub as hub

    @st.cache_resource
    def sbert():
        return SentenceTransformer('sbert')
    sbert_model = sbert()

    @st.cache_resource
    def sbert_embedding(texts):
        return sbert_model.encode(texts)

    @st.cache_data
    def bert_model():
        sbert_embed = np.load('sbert_embed.npy')
        return sbert_embed   
    sbert_embed = bert_model()

    @st.cache_resource
    def use():
        return hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    model = use()

    @st.cache_resource
    def use_embedding(texts):
        return model(texts)
    
    @st.cache_data
    def use_model():
        use_embed = np.load('use_embed.npy')
        return use_embed
    use_embed = use_model()

    selected = option_menu(
        menu_title='Menu',  
        options=["Project", "SBERT", "USE",  "Data", "Visualization"],  
        icons=["book", "gear", 'gear', "code", "eye"],  
        menu_icon="cast",  
        default_index=0,  
        orientation="horizontal",
        styles={
                "container": {"padding": "0!important", "background-color": "#002b36"},
                "icon": {"color": "orange", "font-size": "25px"},
                "nav-link": {
                    "font-size": "25px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "grey"},
            },
    )
    match selected:
        case "Project":
            st.header('About Project')
            st.subheader('This is a final project for Web Application Subject.')
            st.write('The machine learning model in this project uses amazon dataset for training using Nearest Neighbors algorithm. This project also uses word embedding for semantic search to searching the product, semantic search take query input in English text. The result of word embedding process will be used to search for relevant structure and semantic similarity in the query. For word embedding, this project use two models for the embedding process:')
            st.markdown('## 1. **Universal Sentence Encoder**')
            teks = '''The model is trained and optimized for greater-than-word length text, such as sentences, phrases or short paragraphs. 
            It is trained on a variety of data sources and a variety of tasks with the aim of dynamically accommodating a wide variety of natural language understanding tasks. 
            The input is variable length English text and the output is a 512 dimensional vector.'''
            st.write(teks)
            st.write('**Example use**')
            code = '''
            embed = hub.Module("https://kaggle.com/models/google/universal-sentence-encoder/frameworks/TensorFlow1/variations/universal-sentence-encoder/versions/1")
            embeddings = embed([
                            "The quick brown fox jumps over the lazy dog.",
                            "I am a sentence for which I would like to get its embedding"])

            print(session.run(embeddings))'''
            st.code(code, language="python")
            st.write('Source: https://www.kaggle.com/models/google/universal-sentence-encoder')
            st.markdown('## 2. **Sentence BERT (SBERT)**')
            teks1 = '''Sentence-BERT (SBERT) is a modified model of BERT network using siamese and triplet networks. This model maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search.'''
            code1 = '''
            from sentence_transformers import SentenceTransformer
            sentences = ["This is an example sentence", "Each sentence is converted"]

            model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L-6-v3')
            embeddings = model.encode(sentences)
            print(embeddings)'''   
                  
            st.write(teks1)
            st.write('**Example use**')
            st.code(code1, language="python")
            st.write('Source: https://huggingface.co/sentence-transformers/msmarco-MiniLM-L-6-v3')
        case "SBERT":
            
            from sklearn.neighbors import NearestNeighbors
            st.header("Semantic Search using SBERT model")
            number = st.number_input('Input total product', value=5)
            @st.cache_data
            def train_sbert():
                sbert_nn = NearestNeighbors(n_neighbors=number)
                return sbert_nn.fit(sbert_embed)
            sbert_train = train_sbert()

            @st.cache_data
            def sbert_recommend(texts):
                emb = sbert_embedding([texts])
                neighbors = sbert_train.kneighbors(emb, return_distance=False)[0]
                return review_data['ProductId'].iloc[neighbors].tolist()
            
            
            with st.form("Search Product"):
                raw_text = st.text_input("Enter Product Detail Here")
                submit_button = st.form_submit_button(label='Search')
    
                if submit_button:
                    st.info('Result')
                    input = sbert_recommend(raw_text)
                    step = 1
                    for i in input:
                        product_data = review_data.loc[review_data['ProductId'] == i]
                        st.subheader(f"Product {step}")
                        st.write(f"Product ID : {i}")
                        st.write(f"Product Name : {product_data['productName'].iloc[1]}")
                        step += 1
                    st.write('**------------------------------------------------------------------------------------**')    
                    st.subheader('**How to Search the product**')
                    st.write("If you have any trouble to find the product, you can use the link below:")
                    st.write('https://www.amazon.com/dp/ProductId')
                    st.write('**NOTE: Change the ProductId in the link with product id you want to search.**')
                    st.write('')
                    st.write('')
                    st.subheader('**Here is the review of the product**')
                    for i in range(len(input)):
                        st.write(review_data.loc[review_data['ProductId'] == input[i]])   

        case "USE":
            st.header("Semantic Search using USE model")
            number = st.number_input('Input total product', value=5)
            @st.cache_data
            def train_use():
                use_nn = NearestNeighbors(n_neighbors=number)
                return use_nn.fit(use_embed)
            use_train = train_use()

            @st.cache_data
            def use_recommend(texts):
                emb = use_embedding([texts])
                neighbors = use_train.kneighbors(emb, return_distance=False)[0]
                return review_data['ProductId'].iloc[neighbors].tolist()
            
            with st.form("Search Product"):
                raw_text = st.text_input("Enter Product Detail Here")
                submit_button = st.form_submit_button(label='Search')
    
                if submit_button:
                    st.info('Result')
                    input = use_recommend(raw_text)
                    step = 1
                    for i in input:
                        product_data = review_data.loc[review_data['ProductId'] == i]
                        st.subheader(f"Product {step}")
                        st.write(f"Product ID : {i}")
                        st.write(f"Product Name : {product_data['productName'].iloc[1]}")
                        step += 1

                    st.write('**------------------------------------------------------------------------------------**')    
                    st.subheader('**How to Search the product**')
                    st.write("If you have any trouble to find the product, you can use the link below:")
                    st.write('https://www.amazon.com/dp/ProductId')
                    st.write('**NOTE: Change the ProductId in the link with product id you want to search.**')
                    st.write('')
                    st.write('')
                    st.subheader('**Here is the review of the product**')
                    for i in range(len(input)):
                        st.write(review_data.loc[review_data['ProductId'] == input[i]])

        case "Data":
            st.write('This dataset consists of product details from amazon. The details include product and user information (added productName), ratings, and a plain text review. It also includes reviews from all other Amazon categories.')
            st.write('Data Source: https://www.kaggle.com/datasets/aistct/amazonfood?select=amazonFood.csv')
            checkbox = st.checkbox('Show Data')
            if checkbox:
                st.write(df.head(1000))
            list_of_ProductId = df['ProductId'].unique()
            st.write(f'Amount of product\t: {len(list_of_ProductId)}')
            st.write('**Search the product with the link below:**')
            st.write('https://www.amazon.com/dp/productId')
            st.write('Note: Change the ProductId in the link with product id you want to search')

        case 'Visualization':
            tab1,tab2 = st.tabs(['USE Model', 'SBERT Model'])
            
            messages = [
                        # Smartphones
                        "I like my phone",
                        "My phone is not good.",
                        "Your cellphone looks great.",

                        # Weather
                        "Will it snow tomorrow?",
                        "Recently a lot of hurricanes have hit the US",
                        "Global warming is real",

                        # Food and health
                        "An apple a day, keeps the doctors away",
                        "Eating strawberries is healthy",
                        "Is paleo better than keto?",

                        # Asking about age
                        "How old are you?",
                        "what is your age?",
                    ]
            
            def plot_similarity(labels, features, rotation):
                corr = np.inner(features, features)
                sns.set(font_scale=0.8)
                plt.figure(figsize=(8, 6))
                g = sns.heatmap(
                    corr,
                    xticklabels=labels,
                    yticklabels=labels,
                    vmin=0,
                    vmax=1,
                    cmap="YlOrRd")
                g.set_xticklabels(labels, rotation=rotation)
                g.set_title("Semantic Textual Similarity using Universal Sentence Encoder")
                
            with tab1:
                st.write('This is how embedding words process works to measure the semantic similarity in USE model. The density of color in matrix correlation indicates how similar the sentences are.')
                @st.cache_resource
                def run_and_plot(messages_):
                    message_embeddings_ = use_embedding(messages_)
                    plot_similarity(messages_, message_embeddings_, 90)
                    plt.tight_layout()
                    plot_image = plt.gcf()
                    st.pyplot(plot_image)
                    st.write('')
                    st.subheader('**Table of Data in Matrix Correlation**')
                    st.write('Data view of the correlation matrix:')
                    corr = np.inner(message_embeddings_, message_embeddings_)
                    df = pd.DataFrame(corr, columns=messages_, index=messages_)
                    st.write(df)

                run_and_plot(messages)
                
            with tab2:
                st.write('This is how embedding words process works to measure the semantic similarity in SBERT model. The density of color in matrix correlation indicates how similar the sentences are.')
                @st.cache_data
                def correlation_matrix(embeddings):
                    correlation_matrix = util.cos_sim(embeddings, embeddings)
                    return correlation_matrix
                
                def vis_correlation_matrix(correlation_matrix, labels):
                    df = pd.DataFrame(correlation_matrix, columns=labels, index=labels)
                    fig, ax = plt.subplots()
                    sns.set(font_scale=0.8)
                    plt.figure(figsize=(8, 6))
                    g = sns.heatmap(df, 
                                    xticklabels=labels,
                                    yticklabels=labels,
                                    vmin=0, 
                                    vmax=1, 
                                    cmap="YlOrRd",                            
                                    ax=ax)
                    g.set_xticklabels(labels, rotation=90)
                    g.set_title("Semantic Textual Similarity using SBERT")
                    st.pyplot(fig)
                    st.write('')
                    st.subheader('**Table of Data in Matrix Correlation**')
                    st.write('Data view of the correlation matrix:')
                    st.write(df)

                message_embeddings_ = sbert_embedding(messages)
                corr_matrix = correlation_matrix(message_embeddings_)
                vis_correlation_matrix(corr_matrix, messages)
if __name__ == "__main__":
    menu()
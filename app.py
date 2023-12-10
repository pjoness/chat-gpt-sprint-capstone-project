import openai
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import streamlit as st

def initialize_vectorstore(index_name, openai_api_key):
    """
    Initialize a vector store using OpenAI's text-embedding-ada-002 model and Pinecone for similarity search.

    Parameters:
    - index_name (str): The name of the Pinecone index to be created or connected to.
    - openai_api_key (str): OpenAI API key for accessing the text-embedding-ada-002 model.

    Returns:
    - vectorstore (PineconeIndex): A Pinecone vector store object initialized with the specified index and OpenAI model.
    """

    model_name = 'text-embedding-ada-002'

    embed = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=openai_api_key
        )
    
    PINECONE_API_KEY = "045c08fa-f083-4f64-9ed2-377725140fb5"
    
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment="gcp-starter")
    
    index = pinecone.Index(index_name)

    vectorstore = Pinecone.from_existing_index(
        index_name,
        embedding=embed.embed_query)
    
    return vectorstore

def generate_response(query, vectorstore, openai_api_key):
    """
    Generate a response to a given query using OpenAI's language model and a vector store for retrieval.

    Parameters:
    - query (str): The input query for generating a response.
    - vectorstore (PineconeIndex): A Pinecone vector store used for similarity-based retrieval.
    - openai_api_key (str): OpenAI API key for accessing language models.

    Returns:
    None

    Side Effects:
    - Displays the generated response using Streamlit's st.info() function.
    
    Note:
    - This function utilizes OpenAI's language model to generate responses based on the provided query.
    - It employs a vector store for retrieving relevant information.
    - The response is displayed using Streamlit's st.info() function.
    """

    llm = OpenAI(temperature=0.0, openai_api_key=openai_api_key)
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
        )
    
    st.info(qa.run(query))

if __name__ == "__main__":

    st.title('Chatbot App')

    openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

    INDEX_NAME = "langchain-retrieval-augmentation"

    with st.form('my_form'):
        text = st.text_area('Enter text:')
        submitted = st.form_submit_button('Submit')
        if not openai_api_key.startswith('sk-'):
            st.warning('Please enter your OpenAI API key!', icon='⚠')
        if submitted and openai_api_key.startswith('sk-'):
            vectorstore = initialize_vectorstore(
            index_name=INDEX_NAME,
            openai_api_key=openai_api_key)

            generate_response(text, vectorstore, openai_api_key=openai_api_key)

#import packages
import pandas as pd
import faiss,pickle
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer



@st.cache(allow_output_mutation=True)
def get_model( model_id = "multi-qa-mpnet-base-dot-v1"):  
    return SentenceTransformer(model_id)

@st.cache(allow_output_mutation=True)
def load_meta_index():
    #load metadata and faiss index from pickle
    with open('metadata.pkl', 'rb') as handle:
        data = pickle.load(handle)

    with open("faiss_index.pkl",'rb') as h:
        b=pickle.load(h)
        faiss_index=faiss.deserialize_index(b)

    return data,faiss_index

  # function to return similarity and index of metadata 
def vector_search(query,model,index,num_results=3):
  vector=model.encode(list(query))
  D,I =index.search(np.array(vector).astype("float32"),k=num_results)

  return D,I

def main():
    html_text="""<div style="background-color:green;padding:10px">
    <h2 style="color:white;text-align:center;">Query Based Video Timsetamp Retrieval System</h2></div>    
    """
  
    st.markdown(html_text,unsafe_allow_html=True)
    st.sidebar.title("Information :")
    st.sidebar.info("This lists videos URL with timestamps for user based query.")
    st.sidebar.text("For Queries contact :" + "abc@xyz.com")
    
    model=get_model()
    data,faiss_index=load_meta_index() 
    user_query=st.text_area("Enter text", value="Default")

    submit=st.button("Submit")

    if user_query!='Default' and submit:
        D,I=vector_search([user_query],model,faiss_index,3)

        for id_ in I.flatten().tolist():
            st.write("Title of the video :",{data[id_]['title']})
            st.write("Matching text transcript :",data[id_]['text'])
            st.write("Video URL to click :",data[id_]['url']+'&t='+str(int(data[id_]['start'])))
            st.write('######################################################################')
            st.video(data[id_]['url']+'&t='+str(int(data[id_]['start'])))
            st.write('\n\n')
    else:
        st.write("Please enter your query or text in above box ")

if __name__== "__main__":
    main()

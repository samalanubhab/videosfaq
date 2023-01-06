#import packages
import pandas as pd
import faiss,pickle
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer


st.set_page_config(layout="wide")

col1, col2, col3 = st.columns([1,11,4])

with col1:
    st.image("https://assets.kpmg/is/image/kpmg/kpmg-logo-1")

with col2:
    st.write("")
 


with col3:
    st.markdown(
        '<div style="display: flex;">'
        '<a href="https://home.kpmg/xx/en/home.html" target="_blank" style="margin-right: 10px; background-color:#0070c0;color:white;border:none;padding:10px 15px;font-size:14px;font-weight:bold;text-decoration:none;cursor:pointer;border-radius:4px;" class="btn">Contact Us</a>'
        '<a href="https://www.kpmg.com/xx/en/about/index.html" target="_blank" style="margin-left: 10px; background-color:#0070c0;color:white;border:none;padding:10px 15px;font-size:14px;font-weight:bold;text-decoration:none;cursor:pointer;border-radius:4px;" class="btn">About Us</a>'
        '</div>',
        unsafe_allow_html=True,
    )   

st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

st.markdown("""<nav class="navbar navbar-expand-lg navbar-dark" style="background-color: #3498DB;">
  <a class="navbar-brand" href="#"></a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav mx-auto">
      <li class="nav-item active" style="font-size: 50px;">
        <a class="nav-link" href="#">Query Based Video Retrieval System</a>
      </li>       
    </ul>
  </div>
</nav>						
""", unsafe_allow_html=True)



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

    
#     html_text="""<div style="background-color:green;padding:10px">
#     <h2 style="color:white;text-align:center;">Query Based Video Timsetamp Retrieval System</h2></div>    
#     """
  
#     st.markdown(html_text,unsafe_allow_html=True)
#     st.sidebar.title("Information :")
#     st.sidebar.info("This lists videos URL with timestamps for user based query.")
#     st.sidebar.text("For Queries contact :" + "abc@xyz.com")
    
    model=get_model()
    data,faiss_index=load_meta_index() 
    
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

   

#     def icon(icon_name):
#         st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

    local_css("style.css")
    
    col4, col5, col6 = st.columns([0.75,1,0.75])
    
    with col4:
        pass
    with col5:
        st.markdown('**Enter your search query below**')
        user_query = st.text_input("", "")
        
        
        submit=st.button("Submit")
        
       
    with col6:
        pass
    if user_query!='' and submit:

        D,I=vector_search([user_query],model,faiss_index,3)
        i=0
        st.markdown('**Below are your top 3 results**')
        st.write('\n\n')
        
        for id_ in I.flatten().tolist():
        

            i+=1
            st.write(i,".","Title of the video :",{data[id_]['title']},style={"font-size": "500%"}) 

            st.write("Matching text transcript :",data[id_]['text'])
            st.write("Video URL to click :",data[id_]['url']+'&t='+str(int(data[id_]['start'])))  
            
            width = 36
            side = max((100 - width) / 2, 0.01)

            _, container, _ = st.columns([side, width, side])
            container.video(data[id_]['url'],start_time=int(data[id_]['start']))

            st.write('\n\n')
              
    else:
        pass

if __name__== "__main__":
    main()


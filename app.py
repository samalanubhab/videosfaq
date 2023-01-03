#import packages
import pandas as pd
import faiss,pickle
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events

st.set_page_config(layout="wide")

col1, col2, col3 = st.columns([1,11,1])

with col1:
    st.image("logo.jpg")

with col2:
    st.write("")
 


with col3:
    st.write("")

st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

st.markdown("""<nav class="navbar navbar-expand-lg navbar-dark" style="background-color: #3498DB;">
  <a class="navbar-brand" href="https://home.kpmg/xx/en/home.html" target="_blank">Contact Us</a>
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
    
    from bokeh.models.widgets import Button
    from bokeh.models import CustomJS
    from streamlit_bokeh_events import streamlit_bokeh_events

    stt_button = Button(label="Speak", width=100)

    stt_button.js_on_event("button_click", CustomJS(code="""
        var recognition = new webkitSpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;

        recognition.onresult = function (e) {
            var value = "";
            for (var i = e.resultIndex; i < e.results.length; ++i) {
                if (e.results[i].isFinal) {
                    value += e.results[i][0].transcript;
                }
            }
            if ( value != "") {
                document.dispatchEvent(new CustomEvent("GET_TEXT", {detail: value}));
            }
        }
        recognition.start();
        """))

    result = streamlit_bokeh_events(
    stt_button,
    events="GET_TEXT",
    key="listen",
    refresh_on_update=False,
    override_height=75,
    debounce_time=0)

    if result:
        if "GET_TEXT" in result:
            st.write(result.get("GET_TEXT"))
    
    user_query=st.text_area("Enter text", value="Default")
    # Add css to make text bigger
    st.markdown(
    """
    <style>
    textarea {
        font-size: 2rem !important;
    }
    
    </style>
    """,
    unsafe_allow_html=True,
)

    submit=st.button("Submit")

    if user_query!='Default' and submit:
        D,I=vector_search([user_query],model,faiss_index,3)
        i=0
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
        st.write("Please enter your query or text in above box ")

if __name__== "__main__":
    main()


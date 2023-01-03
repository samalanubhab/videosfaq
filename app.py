#import packages
import pandas as pd
import faiss,pickle
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

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
    
    if st.button("Start Recording"):
	    with st.spinner("Recording..."):
		    filename=record()
    
    if st.button("Play Recording"):
        play_record(filename)
        
    user_query=st.text_area("Enter text", value="Default")

    submit=st.button("Submit")

    if user_query!='Default' and submit:
        D,I=vector_search([user_query],model,faiss_index,3)

        for id_ in I.flatten().tolist():
            st.write("Title of the video :",{data[id_]['title']})
            st.write("Matching text transcript :",data[id_]['text'])
            st.write("Video URL to click :",data[id_]['url']+'&t='+str(int(data[id_]['start'])))
            st.write('######################################################################')
            st.video(data[id_]['url'],start_time=int(data[id_]['start']))
            st.write('\n\n')
    else:
        st.write("Please enter your query or text in above box ")

if __name__== "__main__":
    main()


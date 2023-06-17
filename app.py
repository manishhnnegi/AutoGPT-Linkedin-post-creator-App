# Bring in deps
import os 
from apikey import apikey 

import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain , SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

print('success')


os.environ['OPENAI_API_KEY'] = apikey

# App framework
st.title('🦜🔗 GPT LinkedIn Post Creator')
prompt = st.text_input('Plug in your prompt here') 



#create prompt templet
title_template = PromptTemplate(
    input_variables=["topic"],
    template="Write me the Linkedin post title about {topic}"
)



script_template = PromptTemplate(
    input_variables=["title", 'wikipedia_research'],
    template="Write me the Linkedin post script based on this title : {title} while leveraging this wikipedia reserch:{wikipedia_research}"
)



#meomery

title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')







#llm 
llm = OpenAI(temperature=0.9)
#print(prompt_title.format(product="colorful socks"))
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key= 'title', memory= title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory= script_memory)

wiki = WikipediaAPIWrapper()

#sequential_chain = SequentialChain(chains=[title_chain, script_chain], verbose= True, input_variables=['topic'], output_variables=['title','script'])



if prompt:
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title = title, wikipedia_research = wiki_research )

    st.write(title)
    st.write(script)

    with st.expander('Title History'): 
        st.info(title_memory.buffer)

    with st.expander('Script History'): 
        st.info(script_memory.buffer)

    with st.expander('Wikipedia Research'): 
        st.info(wiki_research)



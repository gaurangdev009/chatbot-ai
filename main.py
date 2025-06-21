import os
os.environ["STREAMLIT_WATCH_USE_POLLING"]="true"
# import types
# import torch
# try:
#     if isinstance(torch._classes, types.ModuleType):
#         torch._classes.__path__ = []
# except Exception as e:
#     print("Torch patch failed:", e)
# import fasttext.util
import streamlit as st
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from datetime import datetime , timedelta
import sqlite3
import json

# import anthropic
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph ,END
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_core.documents import Document
from langchain.chains.summarize import load_summarize_chain
# from sentence_transformers import SentenceTransformer
# from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoModelForCausalLM
from transformers import  pipeline, AutoTokenizer, AutoModelForCausalLM
# from langchain.llms import HuggingFacePipeline 
import torch
# import numpy as np
# import fasttext
# from gensim.models import KeyedVectors

# import sys
# import types
# import asyncio
# import torch
# torch.device('cpu')

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import asyncio

def safe_run(coro):
    try:
        return asyncio.get_running_loop().run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)

# try:
#     loop = asyncio.get_running_loop()
# except RuntimeError:
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)


# try:
#     import torch._classes
#     if isinstance(torch._classes, types.ModuleType):
#         # Prevent Streamlit from inspecting torch internals
#         torch._classes.__path__ = []
# except Exception as e:
#     print("Streamlit watcher patch failed:", e)



st.title("Personal AI")
input_text = st.text_input("chat with me")

llm = ChatOllama(model_id = "tinyllama")

output = StrOutputParser()

con = sqlite3.connect("chatbot database")
cr = con.cursor()
cr.execute("""
CREATE TABLE IF NOT EXISTS chat_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    user_message TEXT,
    bot_response TEXT,
    category TEXT,
    mood TEXT,
    theme TEXT,
    goal TEXT
)
""")

memory_folder = "memory"
os.makedirs(memory_folder, exist_ok=True)

prompt_template = PromptTemplate.from_template("Q: {Question}\nA: ...")

category_dynamic_prompt = PromptTemplate.from_template("""
Message: {message}

Category:
""") 

mood_theme_detact = PromptTemplate.from_template("""
Message: "{message}"

respond in json data:
{{
    "mood": "<mood>",
    "theme": "<theme>",
    "goal": "<goal>"
}}                    
""")

custom_template = PromptTemplate.from_template("""
user:{profile}
                                               
input:
{input_text}
""")

# @st.cache_resource
# def load_model():
#     model_name = ORTModelForSequenceClassification.from_pretrained(
#     "distilbert-base-uncased-finetuned-sst-2-english",
#     export=True
#     )
#     tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

#     return ORTModelForSequenceClassification(model_name=model_name , tokenizer = tokenizer)
# fasttext.load_model('cc.en.300.bin')
# fasttext.util.download_model('en', if_exists='ignore' )
@st.cache_resource
def load_model():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v0.6"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    class SimpleLLM:
        def __call__(self, prompt, **kwargs):
            result = generator(prompt, max_new_tokens=256, do_sample=True)
            return result[0]["generated_text"]

    return SimpleLLM()
embedding_model = load_model()

cr.execute("SELECT timestamp, user_message, bot_response FROM chat_logs")
rows = cr.fetchall()

vector_store = None
docs = []
for timestamp, user_meg, bot_res in rows[:100]:
    text = f"user :{user_meg}\n bot:{bot_res}"
    time_date = {"timestamp": timestamp}
    docs.append(Document(page_content=text, metadata=time_date))

@st.cache_resource
def vectorstore_func():
    global vector_store
    if not os.path.exists("./vector_db/index"):
        vector_store = Chroma.from_documents(docs, embedding_model, persist_directory="./vector_db")
    else:
        vector_store = Chroma(persist_directory="./vector_db", embedding_function=embedding_model)
    return vector_store

def my_profile(memory_folder=memory_folder):
    profil = {}
    for file_name in os.listdir(memory_folder):
        if file_name.endswith(".json"):
            file_path = os.path.join(memory_folder, file_name)
            with open(file_path, "r") as f:
                try:
                    data = json.load(f)
                    key = file_name.replace(".json", "")
                    profil[key] = data
                except Exception as e:
                    print(f"error:{e}")
    return profil

def ask_insightful_question(state):
    profile = my_profile()
    recent_summary = vector_summarize()
    input_context = f"""
    User Profile: {json.dumps(profile, indent=2)}
    Weekly Summary: {recent_summary}
    Last User Input: {state['input_text']}
    """
    prompt = f"inputs.Context:\n{input_context}\n\nOutput only the question."
    state_with_prompt = {"prompt": prompt}
    response = safe_run(llm_invoke(state_with_prompt))
    follow_up = response["llm_response"]
    st.markdown(f"Follow-up Question: {follow_up}")
    return {**state, "follow_up_question": follow_up}


def weekly_time(days=7):
    vector_store = vectorstore_func()
    recent_time = datetime.now() - timedelta(days=days)
    all_docs = vector_store.similarity_search("summary", k=100)
    return [doc for doc in all_docs if datetime.fromisoformat(doc.metadata["timestamp"]) >= recent_time]

def vector_summarize():
    weekly_docs = weekly_time()
    if not weekly_docs:
        return "not found chats"
    chain = load_summarize_chain(llm, chain_type="stuff")
    return chain.invoke(weekly_docs)

if st.button("weekly summary"):
    st.info("searching for summary...")
    summary = vector_summarize()
    st.markdown(summary)

def replay_month(days=30):
    vector_store = vectorstore_func()
    current_time = datetime.now() - timedelta(days=days)
    all_docs = vector_store.similarity_search("summary", k=500)
    return [doc for doc in all_docs if datetime.fromisoformat(doc.metadata["timestamp"]) >= current_time]

def vector_replaymonth_summarize():
    replay_docs = replay_month()
    if not replay_docs:
        return "not found chats"
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    return chain.run(replay_docs)

if st.button("Replay my month"):
    st.info("searching for summary...")
    summary = vector_replaymonth_summarize()
    st.markdown(summary)

def prompt_temp(state):
    prompt = prompt_template.format(Question=state["input_text"])
    return {"prompt": prompt, **state}

def llm_invoke(state):
    response = llm.invoke([HumanMessage(content=state["prompt"])])
    content = response.content if hasattr(response, "content") else response
    return {"llm_response": content, **state}


def parsed_output(state):
    text = state["llm_response"]
    output_text = output.parse(text)
    return {"output": output_text, **state}

def category_decide(state):
    prompt = prompt_template.format(Question=state["input_text"])
    response = safe_run(llm_invoke({"prompt":prompt}))
    category = response["llm_response"].strip().lower().replace(" ", "_")
    return {**state , "category": category}

def mood_theme(state):
    prompt = mood_theme_detact.format(message=state["input_text"])
    try:
        response = safe_run(llm_invoke({"prompt":prompt}))
        result = json.loads(response["llm_response"])
    except:
        result = {"mood": "unknown", "theme": "unknown", "goal": "unknown"}
    return {**state,"mood": result["mood"], "theme": result["theme"], "goal": result["goal"]}

def write_my_blog(state):
    memory_files = os.listdir(memory_folder)
    thought = []
    for file in memory_files:
        if file.endswith(".json"):
            file_path = os.path.join(memory_folder, file)
            with open(file_path, "r") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        thought.extend(data)
                except json.JSONDecodeError:
                    continue

    if not thought:
        return {"output_text": "No thoughts found to write a blog.", **state}

    content = "\n\n".join([f"{item['message']}\n{item['response']}" for item in thought])
    prompt = f"write blogs and notes :\n\n{content}"
    response = safe_run(llm_invoke({"prompt":prompt}))
    blog_text = response["llm_response"]

    with open(os.path.join(memory_folder, "blog_output.txt"), "w") as f:
        f.write(blog_text)
    return {"output_text": blog_text, **state}


if st.button("write my blog and notes"):
    st.info("creating blog and notes...")
    blog_notes = write_my_blog({})
    st.markdown(blog_notes["output_text"])

emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base" )
# AutoTokenizer("text-classification", model="j-hartmann/emotion-english-distilroberta-base" )

def ai_therapist(state):
    emotion_detect = emotion_classifier(state["input_text"])[0]
    emotion = emotion_detect['label']
    prompt = f"emotion : {emotion} .\n\n user:{state['input_text']}"
    response = safe_run(llm_invoke({"prompt": prompt}))
    output_text = response["llm_response"]
    return {**state, "output_text": output_text}

def get_responce_profil(state):
    profile = my_profile()
    prompt = custom_template.format(profile=json.dumps(profile, indent=2), input_text=state["input_text"])
    response = safe_run(llm_invoke({"prompt": prompt}))
    content = response["llm_response"]
    return {**state, "output_text": content}



def storage(state):
    required_keys = ["category", "mood", "theme", "goal", "input_text", "output_text"]
    for key in required_keys:
        if key not in state:
            state[key] = "unknown"
            
    file_path = f"{memory_folder}/category.json"
    data = []
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
    data.append({"message": state["input_text"], "response": state["output_text"]})
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
    cr.execute("INSERT INTO chat_logs (timestamp, user_message, bot_response, category, mood, theme, goal) VALUES (?, ?, ?, ?, ?, ?, ?)",
               (str(datetime.now()), state["input_text"], state["output_text"], state["category"], state["mood"], state["theme"], state["goal"]))
    con.commit()
    return state

def final(state):
    state.setdefault("output", state.get("output_text", "No output"))
    st.markdown(f"Bot: {state['output']}")
    st.caption(f"Stored in dynamic category: `{state['category']}` | Mood: `{state['mood']}` | Theme: `{state['theme']}` | Goal: `{state['goal']}`")

graph = StateGraph(state_schema=dict)
graph.set_entry_point("prompt_temp")
graph.add_node("prompt_temp", prompt_temp)
graph.add_node("llm_invoke", llm_invoke)  
graph.add_node("parsed_output", parsed_output)
graph.add_node("category_decide", category_decide)
graph.add_node("mood_theme", mood_theme)
graph.add_node("write_my_blog", write_my_blog)
graph.add_node("ai_therapist", ai_therapist)
graph.add_node("get_responce_profil", get_responce_profil)
graph.add_node("ask_insightful_question", ask_insightful_question)
graph.add_node("storage", storage)
graph.add_node("show_output", final)


graph.add_edge("prompt_temp", "llm_invoke")
graph.add_edge("llm_invoke", "parsed_output")
graph.add_edge("parsed_output", "category_decide")
graph.add_edge("category_decide", "mood_theme")
graph.add_edge("mood_theme", "write_my_blog")
graph.add_edge("write_my_blog", "ai_therapist")
graph.add_edge("ai_therapist", "get_responce_profil")
graph.add_edge("get_responce_profil", "ask_insightful_question")
graph.add_edge("ask_insightful_question", "storage")
graph.add_edge("storage", "show_output")
graph.add_edge("show_output", END)
workflow = graph.compile()

if input_text:
    st.write("Invoking workflow...")
    result_state = workflow.invoke({"input_text": input_text})
    st.write("Workflow done")

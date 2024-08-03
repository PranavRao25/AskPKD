import lancedb
from langchain.text_splitter import *
from langchain.schema.output_parser import StrOutputParser
from langchain_community.llms import HuggingFaceHub
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_core.runnables import RunnableLambda
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
import os
from uuid import uuid4
import streamlit as st


class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_tensor=True).tolist()

    def embed_query(self, text):
        return self.model.encode(text, convert_to_tensor=True).tolist()


class MistralParser:
  """
    Wrapper Class for StrOutputParser Class
    Custom made for Mistral models
  """

  def __init__(self, stopword='Answer:'):
    """
      Initiliases a StrOutputParser as the base parser
    """
    self.parser = StrOutputParser()
    self.stopword = stopword

  def invoke(self, query):
    """
      Invokes the parser and finds the Model response
    """
    ans = self.parser.invoke(query)
    return ans[ans.find(self.stopword)+len(self.stopword):].strip()


class PKDRAG:
  def __init__(self, model, parser, cross_model, embedder, vb, prompt):
    self.chat_model = model
    self.parser = parser
    self.cross_model = cross_model
    self.embedder = embedder
    self.vb = vb
    self.prompt = prompt

  def _context_retrieve(self, question):
    con = self.vb.search(embedder.embed_query(question)).limit(3).to_pandas()['chunk'].tolist()
    c = cross_model.rank(
        query=question,
        documents=con,
        return_documents=True
      )[:len(con) - 2]
    contexts = []
    for q in c:
        contexts.append(q['text'])
    return contexts

  def invoke(self, query):
    context = self._context_retrieve(query)
    self.rag_chain = {'question':RunnablePassthrough(), 'context':RunnableLambda(lambda x: context)} | self.prompt | self.chat_model | self.parser
    return self.rag_chain.invoke(query)


@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")


@st.cache_resource(show_spinner=False)
def load_chat_model():
    return HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        model_kwargs={"temperature": 0.5, "max_length": 64, "max_new_tokens": 512, "query_wrapper_prompt": template}
    )


@st.cache_resource(show_spinner=False)
def load_cross():
    return CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2", max_length=512, device="cpu")


@st.cache_resource(show_spinner=False)
def vb_prep(data):
  chunks = splitter.split_documents(splitter.create_documents(splitter.split_text(data)))
  chunks = [c.page_content for c in chunks]
  chunk_embeddings = [{"chunk": chunk, "vector": embedder.embed_documents(chunk)} for chunk in chunks]
  return db.create_table('pkd', data=chunk_embeddings)


template = '''
        You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question accurately.
        If the question is not related to the context, just answer 'I don't know'.
        Question: {question}
        Context: {context}
        Answer:
        '''
hf = st.secrets['HUGGINGFACEHUB_API_TOKEN']
os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf
embedder = load_embedder()
chat_model = load_chat_model()
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
cross_model = load_cross()
with open('total_data.txt', 'r') as f:
    data = f.read()
db = lancedb.connect('lancedb')
vb = vb_prep(data)
parser = RunnableLambda(MistralParser().invoke)
prompt = ChatPromptTemplate.from_template(template)
bot = PKDRAG(chat_model, parser, cross_model, embedder, vb, prompt)

st.title('Ask PKD')
st.subheader('Ask anything about IIT Palakkad')
if 'conv' not in st.session_state:
  st.session_state['conv'] = {}
if user_prompt := st.chat_input('Ask anything about IIT Palakkad'):
  conv_id = uuid4()
  ai_response = bot.invoke(user_prompt)
  st.session_state['conv'][conv_id] = {
      'user_message': {'role': 'user', 'content': user_prompt},
      'bot_message': {'role': 'assistant', 'content': ai_response}
  }

for conv_id in st.session_state['conv']:
  dic = st.session_state['conv'][conv_id]
  with st.chat_message(dic['user_message']['role']):
    st.markdown(dic['user_message']['content'])
  with st.chat_message(dic['bot_message']['role']):
    st.markdown(dic['bot_message']['content'])

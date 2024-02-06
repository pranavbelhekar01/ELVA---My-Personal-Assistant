from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import requests
from langchain.retrievers import ArxivRetriever
from sklearn.feature_extraction.text import TfidfVectorizer

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_GEN_API_KEY"))

app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Gemini Pro model and start the chat
gemini_model = genai.GenerativeModel("gemini-pro")
gemini_chat = gemini_model.start_chat(history=[])

# Specify the path to the data folder containing PDF files
data_folder = "Data"

def google_search(query):
    # Replace 'YOUR_API_KEY' and 'YOUR_CX' with your API key and Custom Search Engine ID (cx)
    api_key = os.getenv('GOOGLE_API_KEY')
    cx = os.getenv('GOOGLE_CX')

    base_url = "https://www.googleapis.com/customsearch/v1"
    
    params = {
        'key': api_key,
        'cx': cx,
        'q': query,
    }

    response = requests.get(base_url, params=params)
    results = response.json()

    # Extract top 3 search results (title, snippet/description, link, and image link if provided)
    data = []
    for n in range(min(3, len(results.get('items', [])))):
        title = results['items'][n].get('title', 'N/A')
        snippet = results['items'][n].get('snippet', 'N/A')
        link = results['items'][n].get('link', 'N/A')
        image_link = results['items'][n].get('pagemap', {}).get('cse_image', [{}])[0].get('src', 'N/A')

        data.append({
            'title': title,
            'snippet': snippet,
            'link': link,
            'image_link': image_link,
        })

    return data

def youtube_search(query):
    # Replace 'YOUR_YOUTUBE_API_KEY' with your YouTube Data API key
    api_key = os.getenv('YOUTUBE_API_KEY')

    base_url = "https://www.googleapis.com/youtube/v3/search"
    
    params = {
        'key': api_key,
        'part': 'snippet',
        'q': query,
        'type': 'video',
        'maxResults': 3,
    }

    response = requests.get(base_url, params=params)
    results = response.json()

    # Extract top 3 YouTube video results (title, description, and video link)
    data = []
    for item in results.get('items', []):
        title = item['snippet'].get('title', 'N/A')
        description = item['snippet'].get('description', 'N/A')
        video_link = f'https://www.youtube.com/watch?v={item["id"]["videoId"]}'

        data.append({
            'title': title,
            'description': description,
            'video_link': video_link,
        })

    return data

def get_relevant_document(query):
    retriever = ArxivRetriever()
    docs = retriever.get_relevant_documents(query=query, max_docs=3)

    if not docs:
        return {
            'Title': 'No paper found on this topic',
            'Link': 'No link',
            'Content': 'No summary available...'
        }

    contents = [doc.page_content for doc in docs]
    vectorizer = TfidfVectorizer()
    tfidfs = vectorizer.fit_transform(contents)

    query_tfidf = vectorizer.transform([query])
    similarities = tfidfs * query_tfidf.T
    most_similar_idx = similarities.argmax()
    most_similar_doc = docs[most_similar_idx]

    title = most_similar_doc.metadata['Title']
    link = most_similar_doc.metadata['Entry ID']
    content = most_similar_doc.page_content + "..."

    return {
        'Title': title,
        'Link': link,
        'Content': content
    }


def get_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    pdf_files = [os.path.join(data_folder, file) for file in os.listdir(data_folder) if file.endswith(".pdf")]
    raw_text = get_pdf_text(pdf_files)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response_from_chain = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True)

    # Get response from Gemini model
    gemini_response = gemini_chat.send_message(user_question, stream=True)
    response_from_gemini = " ".join([chunk.text for chunk in gemini_response])

    # Get response from Google Search API
    google_search_response = google_search(user_question)

    # Get response from YouTube API
    youtube_response = youtube_search(user_question)

    arxiv_response = get_relevant_document(user_question)

    return {
        "gemini_response": response_from_gemini,
        "qa_chain_response": response_from_chain["output_text"],
        "google_search_response": google_search_response,
        "youtube_response": youtube_response,
        "arxiv_response_title": arxiv_response.get("Title"),
        "arxiv_response_link": arxiv_response.get("Link"),
        "arxiv_response_content": arxiv_response.get("Content")
        # "paper_summary": arxiv_response  # Include the Arxiv response in the result
    }

@app.post('/ask')
async def ask_question(request: Request):
    try:
        data = await request.json()
        user_question = data['question']
        if not user_question:
            raise HTTPException(status_code=400, detail="Empty question")

        # Get responses from different functionalities
        responses = user_input(user_question)

        return JSONResponse(content=responses)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

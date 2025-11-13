import streamlit as st
import json
import requests
# The required library names in requirements.txt are `google-cloud-bigquery` and `google-genai`
from google.cloud import bigquery
from google import genai
from google.genai import types
import uuid # Needed to generate a unique request ID
from google.cloud import logging
from google.cloud import aiplatform

# --- 0. Configuration and Initialization ---
# NOTE: Using st.secrets is recommended for production, but we'll use constants for this example.

# Define your project and model IDs here for BigQuery RAG
PROJECT_ID = "qwiklabs-gcp-03-b295c10c44aa"
DATASET_ID = "rag_dataset"
FAQ_TABLE = f"`{PROJECT_ID}.{DATASET_ID}.embedded_ADS_faqs`"
EMBEDDING_MODEL = f"`{PROJECT_ID}.{DATASET_ID}.ADS_faq_embedding_model`"
GEMINI_MODEL = f"`{PROJECT_ID}.{DATASET_ID}.ADS_gemini_model`"

# Define API Keys for Weather Tool (REPLACE WITH YOUR ACTUAL KEYS)
GEMINI_API_KEY = "AIzaSyCkLF4jio1acMbovFtMSxqDGrc8nP6Mv74"
GOOGLE_API_KEY = "AIzaSyDChIe_z1EufO3lEqlhiP4i9pAfUxzwFvw"

# NWS API requires a user agent
NWS_HEADERS = {
    'User-Agent': '(StreamlitRAGAgent, andy.friedl@nerdery.com)'
}


# --- Cloud Logging Setup --- # <<< NEW: Initialize Logging Client
LOG_NAME = "multi-tool-agent-log"

@st.cache_resource
def init_logging_client():
    """Initializes the Cloud Logging client."""
    try:
        # The client will automatically use Application Default Credentials (ADC)
        return logging.Client(project=PROJECT_ID)
    except Exception as e:
        st.error(f"Failed to initialize Cloud Logging: {e}. Logs will not be sent.")
        return None

logging_client = init_logging_client()
logger = logging_client.logger(LOG_NAME) if logging_client else None


# Initialize BigQuery Client once (Streamlit best practice)
@st.cache_resource
def init_bigquery_client():
    return bigquery.Client(project=PROJECT_ID)


client = init_bigquery_client()


# --- Custom Logging Functions --- # <<< NEW: Functions to Write Logs

def log_user_prompt(prompt: str, request_id: str):
    """Logs the user's submitted prompt."""
    if logger:
        log_payload = {
            "event": "user_prompt_submitted",
            "request_id": request_id,
            "user_id": "streamlit-session-placeholder", # Placeholder for a real user ID
            "prompt": prompt,
        }
        logger.log_struct(
            log_payload,
            severity="INFO",
            # Use a 'global' resource type for applications without a specific GCP resource
            resource={
                "type": "global",
                "labels": {"module_id": "chatbot-frontend"}
            }
        )

def log_final_response(prompt: str, answer: str, tool_used: str, request_id: str):
    """Logs the final response sent back to the user."""
    if logger:
        # Calculate response length (useful for monitoring/billing estimation)
        response_len = len(answer)

        log_payload = {
            "event": "final_response_sent",
            "request_id": request_id, # Key for linking logs
            "tool_used": tool_used,
            "response_length_chars": response_len,
            "prompt_hash": hash(prompt), # Use hash instead of full prompt for the final log
            "response_summary": answer[:100] + "..." if response_len > 100 else answer,
        }
        logger.log_struct(
            log_payload,
            severity="INFO",
            resource={
                "type": "global",
                "labels": {"module_id": "chatbot-frontend"}
            }
        )

# --- 1. Tool Routing Function ---

def route_query(user_query: str) -> str:
    """
    Uses Gemini to determine if the query should use the weather tool or RAG tool.
    Returns 'weather' or 'rag'.
    """
    st.info(f"🧭 Routing query: '{user_query}'")
    try:
        # Initialize GenAI Client
        http_options = types.HttpOptions(api_version='v1beta')
        genai_client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options=http_options,
            vertexai=False
        )

        system_instruction = (
            "You are a routing expert. Your task is to analyze the user's question "
            "and determine the correct tool to use. "
            "If the question explicitly asks for weather, forecast, or is about a location's "
            "atmospheric conditions, return 'weather'. "
            "For all other questions, return 'rag'. "
            "Return ONLY the tool name ('weather' or 'rag') and nothing else."
        )

        config = types.GenerateContentConfig(system_instruction=system_instruction)

        response = genai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[user_query],
            config=config,
        )

        route = response.text.strip().lower()
        if route not in ['weather', 'rag']:
            st.warning(f"Routing failed (unexpected output: '{route}'). Defaulting to 'rag'.")
            return 'rag'

        st.success(f"✅ Routing determined: '{route}'")
        return route

    except Exception as e:
        st.error(f"Error during routing API call: {e}. Defaulting to 'rag'.")
        return 'rag'


# --- 2. RAG Functions (Existing, for ADS FAQs) ---

# Use st.cache_data to cache results from repeated BigQuery calls,
# although for a live RAG query, you might want to adjust this.
def retrieve_relevant_context(user_question: str, top_k: int = 5) -> str:
    """
    Step 1: RAG Retrieval - Get relevant context from BigQuery vector search.
    """
    # ... [Your existing retrieve_relevant_context code block goes here] ...
    cleaned_question = user_question.strip("'").strip('"')
    sql_question = cleaned_question.replace("'", "\\'")

    retrieval_query = f"""
    SELECT
      STRING_AGG(content, '\\n\\n') AS context
    FROM (
      SELECT
        t.content
      FROM
        {FAQ_TABLE} t,
        (
          SELECT ml_generate_embedding_result AS embedding
          FROM ML.GENERATE_EMBEDDING(
            MODEL {EMBEDDING_MODEL},
            (SELECT '{sql_question}' AS content)
          )
        ) q
      ORDER BY ML.DISTANCE(t.embedding, q.embedding, 'COSINE') ASC
      LIMIT {top_k}
    )
    """

    try:
        query_job = client.query(retrieval_query)
        for row in query_job.result():
            return row[0] if row[0] else ""
    except Exception as e:
        raise Exception(f"RAG retrieval failed: {e}")

    return ""


def generate_llm_response(user_question: str, context: str) -> str:
    """
    Step 2: LLM Generation - Feed the context and question to Gemini.
    """
    # ... [Your existing generate_llm_response code block goes here] ...
    # Build the prompt combining user question + RAG context
    prompt = f"""Answer the question based *only* on the following context. If the context does not contain the answer, state that you cannot answer.

Question: {user_question}

Context:
{context}"""

    # Use parameterized query to avoid SQL escaping issues
    generation_query = f"""
    SELECT
      ml_generate_text_result
    FROM ML.GENERATE_TEXT(
      MODEL {GEMINI_MODEL},
      (SELECT @prompt AS prompt),
      STRUCT(
        2048 AS max_output_tokens,
        0.2 AS temperature
      )
    )
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("prompt", "STRING", prompt)
        ]
    )

    try:
        query_job = client.query(generation_query, job_config=job_config)

        for row in query_job.result():
            raw_result = row[0]

            if isinstance(raw_result, str):
                response_json = json.loads(raw_result)
            elif isinstance(raw_result, dict):
                response_json = raw_result
            else:
                raise Exception(f"Unexpected result type: {type(raw_result)}")

            # Extract the text from the JSON structure
            return response_json['candidates'][0]['content']['parts'][0]['text']

    except Exception as e:
        raise Exception(f"LLM generation failed: {e}")

    return "Sorry, I couldn't generate a response."


def get_rag_answer_from_bq(user_question: str) -> str:
    """
    Combined RAG workflow: Retrieve context, then generate answer with LLM.
    """
    with st.spinner("Searching Vector Database and calling Gemini (RAG Mode)..."):
        try:
            # Step 1: Retrieve relevant context using RAG
            context = retrieve_relevant_context(user_question, top_k=5)

            if not context:
                return "No relevant context found in the knowledge base."

            # Step 2: Generate answer using LLM with the retrieved context
            answer = generate_llm_response(user_question, context)

            return answer

        except Exception as e:
            return f"**[RAG ERROR]** {e}"


# --- 3. Weather Tool Functions (New) ---

def get_address_from_query(user_query: str) -> tuple[str | None, bool]:
    # ... [Your existing get_address_from_query code block goes here] ...
    # Initialize GenAI Client
    http_options = types.HttpOptions(api_version='v1beta')
    genai_client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=http_options,
        vertexai=False
    )

    try:
        system_instruction = (
            "You are an expert location extraction tool. Your only task is to extract "
            "the complete, valid, single-line street address from the user's input. "
            "Return ONLY the clean, formatted street address and nothing else. "
            "If no street address is clearly present, return the word 'None'."
        )

        config = types.GenerateContentConfig(system_instruction=system_instruction)

        response = genai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[user_query],
            config=config,
        )

        extracted_address = response.text.strip()

        if extracted_address.lower() == 'none' or not extracted_address:
            return None, False

        return extracted_address, True

    except Exception as e:
        st.error(f"Error during Gemini Address API call: {e}")
        return None, False


def get_lat_lon_from_address(address: str) -> tuple[float | None, float | None]:
    # ... [Your existing get_lat_lon_from_address code block goes here] ...
    geocode_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        'address': address,
        'key': GOOGLE_API_KEY
    }

    try:
        response = requests.get(geocode_url, params=params)
        response.raise_for_status()
        data = response.json()

        if data['status'] == 'OK':
            location = data['results'][0]['geometry']['location']
            lat = location['lat']
            lon = location['lng']
            return lat, lon
        else:
            return None, None

    except requests.exceptions.RequestException as e:
        st.error(f"Error during Geocoding API call: {e}")
        return None, None


def get_nws_forecast(lat: float, lon: float, address: str) -> str:
    # ... [Your existing get_nws_forecast code block goes here] ...
    if lat is None or lon is None:
        return "Sorry, I can't fetch the weather because I couldn't determine the location."

    # Step 1: Get the NWS station/grid point for the coordinates
    points_url = f"https://api.weather.gov/points/{lat},{lon}"

    try:
        response = requests.get(points_url, headers=NWS_HEADERS)
        response.raise_for_status()
        point_data = response.json()

        # The forecast URL is in the 'properties' of the response
        forecast_url = point_data['properties']['forecast']

        # Step 2: Get the detailed forecast
        forecast_response = requests.get(forecast_url, headers=NWS_HEADERS)
        forecast_response.raise_for_status()
        forecast_data = forecast_response.json()

        # Extract and format the forecast for the chatbot response
        periods = forecast_data['properties']['periods']

        # New format for a better Streamlit response
        summary = f"Here is the 7-day weather forecast for **{address}**:\n\n"

        # Limit to the next 5 periods for a cleaner summary
        for period in periods[:5]:
            summary += f"* **{period['name']}**: {period['temperature']}°{period['temperatureUnit']}, {period['shortForecast']}\n"

        return summary

    except requests.exceptions.RequestException as e:
        return f"Sorry, the weather service is currently unavailable. Error: {e}"

    except KeyError:
        return "Sorry, I couldn't find a weather station for that location."

    except Exception as e:
        return f"An unexpected error occurred while fetching the weather: {e}"


def run_forecast_app(user_question: str) -> str:
    """
    Main execution function to run the modularized weather forecast application.
    """
    with st.spinner("Running Weather Tool (Gemini Address Extraction -> Geocoding -> NWS)..."):
        # 1. Use Gemini to convert the question into a clean address
        target_address, address_found = get_address_from_query(user_question)

        if address_found:
            # 2. Use Google Geocoding to get Lat/Lon from the address
            latitude, longitude = get_lat_lon_from_address(target_address)

            # 3. Use NWS to get the weather
            if latitude and longitude:
                return get_nws_forecast(latitude, longitude, target_address)
            else:
                return "Sorry, I couldn't convert the address to geographical coordinates to get the weather."
        else:
            return "I couldn't find a clear address in your question to check the weather. Please try again with a specific location."


# --- 4. Streamlit UI and Main Router ---

def main_tool_router(prompt: str) -> tuple[str, str]: # <<< Change return type
    """
    Routes the user's question to the correct tool and returns the final answer AND the tool used.
    """
    # New Step: Route the query to the correct tool
    tool_to_use = route_query(prompt)

    if tool_to_use == 'weather':
        # Call the Weather Forecast function
        answer = run_forecast_app(prompt)
    else:
        # Call the BigQuery RAG function (for ADS FAQs)
        answer = get_rag_answer_from_bq(prompt)

    return answer, tool_to_use # <<< Return both the answer and the tool


# --- Streamlit Frontend ---
st.set_page_config(page_title="Multi-Tool RAG Agent (ADS & Weather)", layout="wide")
st.title("🤖 Multi-Tool Chatbot Agent")
st.subheader("Uses BigQuery RAG for ADS FAQs and external APIs for Weather Forecast.")

# Initialize Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Existing Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle User Input
# Handle User Input
if prompt := st.chat_input("Ask about ADS, or ask for the weather at a location..."):
    # --- Logging and Processing Flow ---

    # Generate a unique ID to link the prompt log to the response log
    request_id = str(uuid.uuid4())

    # 1. Add and display the user's message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Log the USER PROMPT SUBMISSION # <<< NEW: Log 1
    log_user_prompt(prompt, request_id)

    # 3. Process the input through the main router
    with st.chat_message("assistant"):
        # Use a placeholder while processing to give instant feedback
        placeholder = st.empty()

        # Get the final answer and the tool used (updated main_tool_router)
        final_answer, tool_used = main_tool_router(prompt) # <<< Line changed to capture tool_used

        # 4. Log the FINAL RESPONSE SENT # <<< NEW: Log 2
        log_final_response(prompt, final_answer, tool_used, request_id)

        # Display the final answer
        placeholder.markdown(final_answer)

    # 5. Add the assistant's response to the chat history
    st.session_state.messages.append({"role": "assistant", "content": final_answer})
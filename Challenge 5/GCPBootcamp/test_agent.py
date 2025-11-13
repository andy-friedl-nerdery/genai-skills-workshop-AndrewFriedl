import pytest
from unittest.mock import MagicMock
import json

# Assuming your main application file is named 'app.py' or similar.
# This import will only work if your file is named 'app.py'. Adjust as needed.
import app


# --- Setup Fixtures (Shared Mock Data) ---

# We don't need to patch the client directly, but the object it returns.

@pytest.fixture
def mock_genai_client(mocker):
    """Mocks the google-genai.Client instance and returns the mock object."""
    # Mock the client instance creation
    MockClient = mocker.patch('app.genai.Client')
    mock_instance = MockClient.return_value
    return mock_instance


@pytest.fixture
def mock_bigquery_client(mocker):
    """Mocks the global BigQuery client instance."""
    return mocker.patch('app.client')


@pytest.fixture
def mock_requests_get(mocker):
    """Mocks the requests.get method used by the weather tool."""
    return mocker.patch('app.requests.get')


# --- NEW: Streamlit Mocking Fixture ---
@pytest.fixture(autouse=True)
def mock_streamlit_dependencies(mocker):
    """
    Mocks necessary Streamlit functions to allow module import and function calls
    without the Streamlit runtime environment.
    """
    # CRUCIAL: Mocks st.cache_resource so it runs the decorated function directly,
    # allowing BQ and Logging clients to initialize for tests.
    mocker.patch('app.st.cache_resource', side_effect=lambda func: func)

    # Mocks all UI/logging functions called outside the main chat loop
    mocker.patch('app.st.set_page_config')
    mocker.patch('app.st.title')
    mocker.patch('app.st.subheader')
    mocker.patch('app.st.info')
    mocker.patch('app.st.warning')
    mocker.patch('app.st.error')
    mocker.patch('app.st.spinner', side_effect=lambda x: (yield))
    mocker.patch('app.st.empty', return_value=MagicMock())


# --- Test Suite for Core Functions ---
class TestToolRoutingAndRAG:


    def test_route_query_weather(self, mock_genai_client):
        """Tests that a weather-related query is correctly routed."""
        # Setup the mock response text
        mock_response = MagicMock()
        mock_response.text = "weather"
        mock_genai_client.models.generate_content.return_value = mock_response

        query = "What is the forecast in London tomorrow?"
        result = app.route_query(query)

        assert result == 'weather'

        # Verify call arguments
        args, kwargs = mock_genai_client.models.generate_content.call_args
        assert "routing expert" in kwargs['config'].system_instruction

    def test_route_query_rag(self, mock_genai_client):
        """Tests that an ADS-related query is correctly routed to RAG."""
        # Setup the mock response text
        mock_response = MagicMock()
        mock_response.text = "rag"
        mock_genai_client.models.generate_content.return_value = mock_response

        query = "What is the policy on data retention?"
        result = app.route_query(query)

        assert result == 'rag'

    def test_retrieve_relevant_context_success(self, mock_bigquery_client):
        """Tests RAG retrieval with a successful BigQuery response."""

        # Mock the BigQuery job result to return the context string
        mock_row = MagicMock()
        mock_row.__getitem__.return_value = "Context from the knowledge base."
        mock_job = MagicMock()
        mock_job.result.return_value = [mock_row]

        mock_bigquery_client.query.return_value = mock_job

        query = "What are the core features of the system?"
        context = app.retrieve_relevant_context(query)

        assert "knowledge base" in context

        # Assert that the correct SQL components were used
        sql_called = mock_bigquery_client.query.call_args[0][0]
        assert app.FAQ_TABLE in sql_called
        assert app.EMBEDDING_MODEL in sql_called

    def test_generate_llm_response_success(self, mock_bigquery_client):
        """Tests LLM generation with a successful BigQuery response."""

        # Define the mocked JSON response structure from the BQ GENERATE_TEXT model
        mock_json_response = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "The answer is based on the provided context."}]
                }
            }]
        }
        mock_raw_result = json.dumps(mock_json_response)

        # Mock the BigQuery job result
        mock_row = MagicMock()
        mock_row.__getitem__.return_value = mock_raw_result
        mock_job = MagicMock()
        mock_job.result.return_value = [mock_row]

        mock_bigquery_client.query.return_value = mock_job

        question = "What is X?"
        context = "X is defined as Y."

        answer = app.generate_llm_response(question, context)

        assert "answer is based on" in answer

        # Verify the prompt was correctly constructed and passed via query parameters
        query_config = mock_bigquery_client.query.call_args[1]['job_config']
        prompt_param = query_config.query_parameters[0]

        assert prompt_param.name == 'prompt'
        assert question in prompt_param.value
        assert context in prompt_param.value


class TestWeatherTool:

    def test_get_address_from_query_success(self, mock_genai_client):
        """Tests Gemini's ability to extract an address from a query."""
        # Setup the mock response text
        mock_response = MagicMock()
        mock_response.text = "123 Main St, Anytown, USA"
        mock_genai_client.models.generate_content.return_value = mock_response

        query = "Can I get the weather for 123 Main St, Anytown, USA?"
        address, found = app.get_address_from_query(query)

        assert address == "123 Main St, Anytown, USA"
        assert found is True

        # Verify the specific instruction was used
        args, kwargs = mock_genai_client.models.generate_content.call_args
        assert "expert location extraction tool" in kwargs['config'].system_instruction

    def test_get_lat_lon_success(self, mock_requests_get):
        """Tests successful geocoding API call."""

        # Configure the mock response for requests.get
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'status': 'OK',
            'results': [{'geometry': {'location': {'lat': 40.7128, 'lng': -74.0060}}}]
        }
        mock_requests_get.return_value = mock_response

        address = "New York, NY"
        lat, lon = app.get_lat_lon_from_address(address)

        assert lat == 40.7128
        assert lon == -74.0060

    def test_get_nws_forecast_success(self, mock_requests_get):
        """Tests successful fetching and formatting of the NWS forecast."""

        # Mocking the two required API calls (points and then forecast)

        # Call 1: The /points API response
        mock_points_response = MagicMock()
        mock_points_response.status_code = 200
        mock_points_response.json.return_value = {
            'properties': {'forecast': 'https://api.weather.gov/gridpoints/ABC/123/forecast'}
        }

        # Call 2: The /forecast API response
        mock_forecast_response = MagicMock()
        mock_forecast_response.status_code = 200
        mock_forecast_response.json.return_value = {
            'properties': {'periods': [
                {'name': 'Today', 'temperature': 75, 'temperatureUnit': 'F', 'shortForecast': 'Sunny'},
                {'name': 'Tonight', 'temperature': 55, 'temperatureUnit': 'F', 'shortForecast': 'Clear'},
            ]}
        }

        # Sequence the responses for the two requests.get calls
        mock_requests_get.side_effect = [mock_points_response, mock_forecast_response]

        lat = 33.0
        lon = -80.0
        address = "Charleston, SC"

        forecast = app.get_nws_forecast(lat, lon, address)

        # Check for the summary header and correct address
        assert "7-day weather forecast for **Charleston, SC**" in forecast

        # Check for the formatted period data
        assert "* **Today**: 75°F, Sunny" in forecast
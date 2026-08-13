"""
Streamlit Web Interface for Natural Language to SQL Application
Main entry point for the web-based NL2SQL application
"""

# SQLite compatibility fix for Streamlit Cloud
import sys
try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
except ImportError:
    pass

import streamlit as st
import logging
import os
import json
import pandas as pd
import time
from typing import Dict, Any, Optional
import traceback
from dotenv import load_dotenv

from database_manager import DatabaseManager
from crew_setup import NL2SQLCrew
from agents import NL2SQLAgents
from tasks import NL2SQLTasks

# Load environment variables from .env file when running locally.
# On Streamlit Cloud/Railway, environment variables are supplied by the platform.
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="NL2SQL CrewAI Application",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .sql-query {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .result-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def extract_sql_from_text(text: str) -> Optional[str]:
    """Extract SQL query from text using improved regex patterns"""
    import re

    sql_patterns = [
        r'```sql\s*(.*?)\s*```',
        r'```SQL\s*(.*?)\s*```',
        r'```\s*(SELECT.*?)\s*```',
        r'```\s*(select.*?)\s*```',
        r'(?:SQL[:\s]+|Query[:\s]+|Generated Query[:\s]+)?\s*```\s*(SELECT.*?)\s*```',
        r'(?:SQL[:\s]+|Query[:\s]+|Generated Query[:\s]+)?\s*```\s*(select.*?)\s*```',
        r'(?:^|\n)(?:SQL[:\s]+|Query[:\s]+|Generated Query[:\s]+)\s*(SELECT.*?)(?=\n\n|\n[A-Z]|\n#|\nExplanation|\nThe|\nThis|\Z)',
        r'(?:^|\n)(?:SQL[:\s]+|Query[:\s]+|Generated Query[:\s]+)\s*(select.*?)(?=\n\n|\n[A-Z]|\n#|\nExplanation|\nThe|\nThis|\Z)',
        r'(?:^|\n)\s*(SELECT\s+(?:DISTINCT\s+)?.*?FROM\s+\w+.*?)(?=\n\n|\n[A-Z]|\n#|\nExplanation|\nThe|\nThis|\Z)',
        r'(?:^|\n)\s*(select\s+(?:distinct\s+)?.*?from\s+\w+.*?)(?=\n\n|\n[A-Z]|\n#|\nExplanation|\nThe|\nThis|\Z)',
        r'(SELECT.*?;)',
        r'(select.*?;)',
        r'\b(SELECT\s+.*?FROM\s+\w+.*?)(?:\s|$)',
        r'\b(select\s+.*?from\s+\w+.*?)(?:\s|$)',
    ]

    for pattern in sql_patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            sql_query = match.strip()
            sql_query = re.sub(r'\s+', ' ', sql_query)
            sql_query = sql_query.replace('```', '').strip()
            if (sql_query.upper().startswith('SELECT') and
                len(sql_query) > 10 and
                'FROM' in sql_query.upper()):
                return sql_query
    return None

def extract_interpretation_from_text(text: str, sql_query: Optional[str] = None) -> str:
    """Extract interpretation part from text, removing SQL code blocks"""
    import re
    if not text:
        return ""
    if sql_query:
        text = re.sub(r'```sql.*?```', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'```SQL.*?```', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = text.replace(sql_query, '')
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def format_sql_query(sql_query: str) -> str:
    """Format SQL query with proper line breaks and indentation"""
    import re
    if not sql_query:
        return sql_query
    sql_query = re.sub(r'\s+', ' ', sql_query.strip())
    formatted_sql = sql_query
    formatted_sql = re.sub(r'\bSELECT\b', '\nSELECT', formatted_sql, flags=re.IGNORECASE)
    formatted_sql = re.sub(r'\bFROM\b', '\nFROM', formatted_sql, flags=re.IGNORECASE)
    formatted_sql = re.sub(r'\bWHERE\b', '\nWHERE', formatted_sql, flags=re.IGNORECASE)
    formatted_sql = re.sub(r'\b(LEFT\s+JOIN|RIGHT\s+JOIN|INNER\s+JOIN|OUTER\s+JOIN|JOIN)\b', r'\n\1', formatted_sql, flags=re.IGNORECASE)
    formatted_sql = re.sub(r'\bON\b', '\n    ON', formatted_sql, flags=re.IGNORECASE)
    formatted_sql = re.sub(r'\bGROUP\s+BY\b', '\nGROUP BY', formatted_sql, flags=re.IGNORECASE)
    formatted_sql = re.sub(r'\bHAVING\b', '\nHAVING', formatted_sql, flags=re.IGNORECASE)
    formatted_sql = re.sub(r'\bORDER\s+BY\b', '\nORDER BY', formatted_sql, flags=re.IGNORECASE)
    formatted_sql = re.sub(r'\bLIMIT\b', '\nLIMIT', formatted_sql, flags=re.IGNORECASE)
    formatted_sql = re.sub(r'\bUNION(\s+ALL)?\b', r'\nUNION\1', formatted_sql, flags=re.IGNORECASE)
    formatted_sql = re.sub(r'\b(AND|OR)\b(?=.*)', r'\n    \1', formatted_sql, flags=re.IGNORECASE)
    formatted_sql = re.sub(r'\n\s*\n', '\n', formatted_sql).strip()
    if formatted_sql.startswith('\n'):
        formatted_sql = formatted_sql[1:]
    return formatted_sql

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    if 'crew' not in st.session_state:
        st.session_state.crew = None
    if 'connection_status' not in st.session_state:
        st.session_state.connection_status = False
    if 'schema_cached' not in st.session_state:
        st.session_state.schema_cached = False
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'current_schema' not in st.session_state:
        st.session_state.current_schema = None
    if 'current_ai_result' not in st.session_state:
        st.session_state.current_ai_result = None
    if 'feedback_examples' not in st.session_state:
        st.session_state.feedback_examples = []
    if 'user_feedback' not in st.session_state:
        st.session_state.user_feedback = {}
    if 'nba_database_connected' not in st.session_state:
        st.session_state.nba_database_connected = False

def sidebar_database_config():
    """Render database configuration sidebar"""
    st.sidebar.header("🗄️ Database Configuration")
    st.sidebar.subheader("Quick Start")
    if st.sidebar.button("🏀 Try Sample NBA Dataset", type="primary"):
        connect_sample_database()

    if st.session_state.connection_status and st.session_state.nba_database_connected:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🏀 Sample Questions")
        st.sidebar.markdown("""
        **Try these NBA queries:**
        - How many teams are in the NBA?
        - List all teams from California
        - Who are the players with 'James' in their name?
        - Show me the Boston Celtics team details
        - How many players are in the database?
        - Which teams were founded before 1950?
        """)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Connect Your Database")
    db_type = st.sidebar.selectbox("Database Type", ["SQLite", "PostgreSQL", "MySQL"])
    connection_params = {}

    if db_type == "SQLite":
        uploaded_file = st.sidebar.file_uploader("Upload SQLite Database", type=['sqlite', 'db', 'sqlite3'])
        if uploaded_file is not None:
            temp_path = f"/tmp/{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            connection_params = {'db_type': 'sqlite', 'file_path': temp_path}
            st.sidebar.success(f"File uploaded: {uploaded_file.name}")
        file_path = st.sidebar.text_input("Or enter SQLite file path")
        if file_path and not uploaded_file:
            connection_params = {'db_type': 'sqlite', 'file_path': file_path}
    elif db_type in ["PostgreSQL", "MySQL"]:
        host = st.sidebar.text_input("Host", value="localhost")
        port = st.sidebar.number_input("Port", value=5432 if db_type == "PostgreSQL" else 3306, min_value=1, max_value=65535)
        database = st.sidebar.text_input("Database Name")
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if all([host, database, username, password]):
            connection_params = {'db_type': db_type.lower(), 'host': host, 'port': port, 'database': database, 'username': username, 'password': password}

    if st.sidebar.button("Connect to Database"):
        if connection_params:
            connect_to_database(connection_params)
        else:
            st.sidebar.error("Please fill in all required connection parameters")

    if st.session_state.connection_status:
        if st.session_state.nba_database_connected:
            st.sidebar.success("✅ NBA Sample Database Connected")
        else:
            st.sidebar.success("✅ Database Connected")
        try:
            tables = st.session_state.db_manager.get_table_names()
            st.sidebar.info(f"📊 {len(tables)} tables found")
        except Exception as e:
            st.sidebar.warning(f"Could not retrieve table count: {str(e)}")
    else:
        st.sidebar.warning("❌ Database Not Connected")
    return connection_params

def create_crew():
    """Create the NL2SQL Crew using the configured Groq model."""
    return NL2SQLCrew(
        st.session_state.db_manager,
        model_name=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    )

def connect_to_database(params: Dict[str, Any]):
    """Connect to database with given parameters"""
    try:
        with st.spinner("Connecting to database..."):
            success = st.session_state.db_manager.connect(**params)
            if success:
                st.session_state.connection_status = True
                st.session_state.nba_database_connected = False
                st.session_state.crew = create_crew()
                st.sidebar.success("Database connected successfully!")
                st.session_state.schema_cached = False
                st.session_state.current_schema = None
                analyze_schema()
            else:
                st.sidebar.error("Failed to connect to database")
                st.session_state.connection_status = False
    except Exception as e:
        st.sidebar.error(f"Database connection error: {str(e)}")
        st.session_state.connection_status = False

def connect_sample_database():
    """Connect to the sample NBA database"""
    try:
        with st.spinner("Connecting to sample NBA database..."):
            nba_db_path = os.path.join(os.getcwd(), "nba.sqlite")
            if not os.path.exists(nba_db_path):
                st.sidebar.error("Sample NBA database not found")
                return
            success = st.session_state.db_manager.connect(db_type='sqlite', file_path=nba_db_path)
            if success:
                st.session_state.connection_status = True
                st.session_state.nba_database_connected = True
                st.session_state.crew = create_crew()
                st.sidebar.success("Connected to NBA database!")
                st.session_state.schema_cached = False
                st.session_state.current_schema = None
                analyze_schema()
            else:
                st.sidebar.error("Failed to connect to sample database")
    except Exception as e:
        st.sidebar.error(f"Sample database connection error: {str(e)}")

def analyze_schema():
    """Analyze database schema"""
    if not st.session_state.connection_status or not st.session_state.crew:
        return
    try:
        with st.spinner("Analyzing database schema..."):
            db_type = st.session_state.db_manager.database_type or "Unknown"
            schema_analysis = st.session_state.crew.analyze_schema(db_type, "connected_database")
            st.session_state.current_schema = schema_analysis
            st.session_state.schema_cached = True
            st.success("Schema analysis completed!")
    except Exception as e:
        st.error(f"Schema analysis failed: {str(e)}")

def check_api_configuration():
    """Check Groq API configuration without exposing the key in the UI."""
    api_key = os.getenv("GROQ_API_KEY")
    selected_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    if not api_key:
        return None, None
    st.session_state.selected_model = selected_model
    return api_key, selected_model

def render_schema_viewer():
    """Render database schema viewer"""
    st.subheader("📊 Database Schema")
    if not st.session_state.connection_status:
        st.warning("Please connect to a database first")
        return
    if not st.session_state.schema_cached:
        if st.button("Analyze Schema"):
            analyze_schema()
        return
    try:
        tables = st.session_state.db_manager.get_table_names()
        selected_table = st.selectbox("Select Table to View", ["All Tables"] + tables)
        if selected_table == "All Tables":
            st.write("### Database Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Tables", len(tables))
            with col2:
                try:
                    total_rows = sum(st.session_state.db_manager.get_table_stats(table).get('row_count', 0) for table in tables)
                    st.metric("Total Rows", f"{total_rows:,}")
                except Exception:
                    st.metric("Total Rows", "N/A")
            with col3:
                st.metric("Database Type", st.session_state.db_manager.database_type.upper())
            st.write("### Tables")
            for table in tables:
                with st.expander(f"📋 {table}"):
                    try:
                        stats = st.session_state.db_manager.get_table_stats(table)
                        schema = st.session_state.db_manager.get_table_schema(table)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Rows:** {stats.get('row_count', 'N/A'):,}")
                            st.write(f"**Columns:** {len(schema.get('columns', []))}")
                        with col2:
                            if schema.get('primary_keys'):
                                st.write(f"**Primary Keys:** {', '.join(schema['primary_keys'])}")
                            if schema.get('foreign_keys'):
                                st.write(f"**Foreign Keys:** {len(schema['foreign_keys'])}")
                        if schema.get('columns'):
                            st.write("**Columns:**")
                            for col in schema['columns'][:5]:
                                st.write(f"- {col['name']}: {col['type']}")
                            if len(schema['columns']) > 5:
                                st.write(f"... and {len(schema['columns']) - 5} more columns")
                    except Exception as e:
                        st.error(f"Error loading table info: {str(e)}")
        else:
            st.write(f"### Table: {selected_table}")
            try:
                schema = st.session_state.db_manager.get_table_schema(selected_table)
                stats = st.session_state.db_manager.get_table_stats(selected_table)
                sample_data = st.session_state.db_manager.get_sample_data(selected_table, limit=5)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Rows", f"{stats.get('row_count', 'N/A'):,}")
                with col2:
                    st.metric("Columns", len(schema.get('columns', [])))
                with col3:
                    st.metric("Primary Keys", len(schema.get('primary_keys', [])))
                st.write("#### Columns")
                if schema.get('columns'):
                    columns_df = pd.DataFrame([{'Column': col['name'], 'Type': col['type'], 'Nullable': 'Yes' if col['nullable'] else 'No', 'Default': col.get('default', 'None')} for col in schema['columns']])
                    st.dataframe(columns_df, use_container_width=True)
                if schema.get('primary_keys') or schema.get('foreign_keys'):
                    st.write("#### Constraints")
                    if schema.get('primary_keys'):
                        st.write(f"**Primary Keys:** {', '.join(schema['primary_keys'])}")
                    if schema.get('foreign_keys'):
                        st.write("**Foreign Keys:**")
                        for fk in schema['foreign_keys']:
                            st.write(f"- {fk['column']} → {fk['referenced_table']}.{fk['referenced_column']}")
                st.write("#### Sample Data")
                if sample_data['success'] and sample_data['data']:
                    st.dataframe(pd.DataFrame(sample_data['data']), use_container_width=True)
                else:
                    st.info("No sample data available")
            except Exception as e:
                st.error(f"Error loading table details: {str(e)}")
    except Exception as e:
        st.error(f"Error loading schema: {str(e)}")

def render_query_interface():
    """Render natural language query interface"""
    st.subheader("💬 Natural Language Query")
    if not st.session_state.connection_status:
        st.warning("Please connect to a database first")
        return
    if not os.getenv("GROQ_API_KEY"):
        st.warning("Service is not available. Please configure GROQ_API_KEY.")
        return
    question = st.text_area("Enter your question", placeholder="e.g., How many teams are in the NBA?", height=100)
    col1, col2 = st.columns([2, 1])
    with col1:
        process_query = st.button("🚀 Process Query", type="primary")
    with col2:
        skip_schema = st.checkbox("⚡ Fast Mode", value=False, help="Skip schema analysis for faster processing (uses cached schema)")
    if process_query:
        if not question.strip():
            st.error("Please enter a question")
            return
        process_natural_language_query(question, use_full_workflow=True, show_metrics=False, skip_schema=skip_schema)
    if 'current_ai_result' in st.session_state and st.session_state.current_ai_result:
        ai_result = st.session_state.current_ai_result
        display_query_results(ai_result['result'], ai_result['question'], ai_result['processing_time'], ai_result['show_metrics'])

def process_natural_language_query(question: str, use_full_workflow: bool, show_metrics: bool, skip_schema: bool = False):
    """Process natural language query through CrewAI"""
    if not st.session_state.crew:
        st.error("Crew not initialized. Please check database connection.")
        return
    try:
        with st.spinner("Processing your query..."):
            start_time = time.time()
            db_type = st.session_state.db_manager.database_type or "Unknown"
            result = st.session_state.crew.process_query(question, use_full_workflow, db_type, "connected_database", skip_schema)
            processing_time = time.time() - start_time
        st.session_state.current_ai_result = {'question': question, 'result': result, 'processing_time': processing_time, 'show_metrics': show_metrics, 'timestamp': time.time()}
        st.session_state.query_history.append({'question': question, 'timestamp': time.time(), 'result': result, 'processing_time': processing_time})
    except Exception as e:
        st.error(f"Query processing failed: {str(e)}")
        st.expander("Error Details", expanded=False).code(traceback.format_exc())

# The remaining UI/result/history helpers are unchanged from the original application.
# They are intentionally kept in this file so the Groq migration only changes the LLM layer.
def display_query_results(result: Dict[str, Any], question: str, processing_time: float, show_metrics: bool):
    """Display query processing results."""
    st.subheader("📊 Query Results")
    if not result.get('success', False):
        st.error(f"❌ Query processing failed: {result.get('error', 'Unknown error')}")
        return
    st.success(f"✅ Query processed successfully in {processing_time:.2f} seconds")
    sql_query = result.get('sql_query') or extract_sql_from_text(result.get('raw_output', ''))
    if sql_query:
        st.write("#### 🔍 Generated SQL Query")
        st.code(format_sql_query(sql_query), language='sql')
        st.write("#### 📊 Query Results")
        execution_result = execute_sql_query(sql_query, return_results=True)
        if execution_result:
            result['execution_result'] = execution_result
    else:
        st.warning("⚠️ Could not extract SQL query from the response")
        with st.expander("🔍 View Raw Response"):
            st.text(result.get('raw_output', ''))
    if show_metrics:
        show_performance_metrics(result, processing_time)

def execute_sql_query(sql_query: str, return_results: bool = False):
    try:
        with st.spinner("Executing SQL query..."):
            query_result = st.session_state.db_manager.execute_query(sql_query)
        if query_result['success']:
            row_count = query_result.get('row_count', 0)
            st.success(f"✅ Query executed successfully - {row_count} rows returned")
            if query_result['data']:
                df = pd.DataFrame(query_result['data'])
                st.dataframe(df, use_container_width=True)
                st.info(f"📋 Showing {len(df)} rows × {len(df.columns)} columns")
            return query_result if return_results else None
        st.error(f"❌ Query execution failed: {query_result.get('error', 'Unknown error')}")
        return query_result if return_results else None
    except Exception as e:
        st.error(f"❌ Error executing SQL query: {str(e)}")
        return {"success": False, "error": str(e), "data": None, "row_count": 0} if return_results else None

def show_performance_metrics(result: Dict[str, Any], processing_time: float):
    st.write("#### Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Processing Time", f"{processing_time:.2f}s")
    with col2:
        st.metric("Workflow Type", result.get('workflow_type', 'Unknown'))
    with col3:
        positive_count = sum(1 for feedback in st.session_state.user_feedback.values() if feedback == "positive")
        st.metric("Queries Liked", positive_count)
    with col4:
        negative_count = sum(1 for feedback in st.session_state.user_feedback.values() if feedback == "negative")
        st.metric("Queries Disliked", negative_count)

def render_query_history():
    st.subheader("📜 Query History")
    if not st.session_state.query_history:
        st.info("No queries processed yet")
        return
    if st.button("🗑️ Clear History"):
        st.session_state.query_history = []
        st.rerun()
    for i, entry in enumerate(reversed(st.session_state.query_history)):
        with st.expander(f"Query {len(st.session_state.query_history) - i}: {entry['question'][:50]}..."):
            st.write(f"**Question:** {entry['question']}")
            st.write(f"**Time:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(entry['timestamp']))}")
            st.write(f"**Processing Time:** {entry['processing_time']:.2f}s")
            if entry['result'].get('sql_query'):
                st.code(format_sql_query(entry['result']['sql_query']), language='sql')
            if entry['result'].get('success'):
                st.success("✅ Query processed successfully")
            else:
                st.error(f"❌ Query processing failed: {entry['result'].get('error', 'Unknown error')}")

def main():
    initialize_session_state()
    load_feedback_from_file()
    st.title("🤖 Natural Language to SQL")
    st.markdown("Convert natural language to SQL using Groq + CrewAI")
    api_key, model = check_api_configuration()
    with st.sidebar:
        sidebar_database_config()
        st.markdown("---")
        st.header("ℹ️ Application Info")
        st.info(f"""
        This application uses CrewAI with 3 specialized agents:
        - **Schema Analyst**: Analyzes database structure
        - **SQL Generator**: Converts NL to SQL
        - **SQL Evaluator**: Validates and executes queries

        **LLM provider:** Groq
        **Model:** `{model or os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')}`
        """)
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Query", "📊 Schema", "📜 History", "⚡ Performance"])
    with tab1:
        render_query_interface()
    with tab2:
        render_schema_viewer()
    with tab3:
        render_query_history()
    with tab4:
        st.subheader("⚡ Performance Dashboard")
        if st.session_state.crew:
            metrics = st.session_state.crew.get_performance_metrics()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Queries", metrics.get('total_queries', 0))
            with col2:
                st.metric("Avg. Processing Time", f"{metrics.get('average_execution_time', 0):.2f}s")
            with col3:
                st.metric("Queries Liked", sum(1 for f in st.session_state.user_feedback.values() if f == "positive"))
            with col4:
                st.metric("Queries Disliked", sum(1 for f in st.session_state.user_feedback.values() if f == "negative"))
            if st.button("🔄 Reset Metrics"):
                st.session_state.crew.reset_metrics()
                st.success("Metrics reset successfully")
                st.rerun()
        else:
            st.info("Connect to a database to view performance metrics")

# Keep feedback helpers available to the existing workflow.
def save_feedback_to_file():
    try:
        with open("feedback_examples.json", "w") as f:
            json.dump({'examples': st.session_state.feedback_examples, 'last_updated': time.time()}, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving feedback to file: {str(e)}")

def load_feedback_from_file():
    try:
        feedback_file = "feedback_examples.json"
        if os.path.exists(feedback_file):
            with open(feedback_file, 'r') as f:
                st.session_state.feedback_examples = json.load(f).get('examples', [])
    except Exception as e:
        logger.error(f"Error loading feedback from file: {str(e)}")

if __name__ == "__main__":
    main()

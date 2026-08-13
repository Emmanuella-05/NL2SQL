"""
CrewAI Agents for Natural Language to SQL Application
Defines specialized agents for different aspects of the NL2SQL process
"""

import logging
from typing import List, Dict, Any
import os

from crewai import Agent
from langchain_groq import ChatGroq
from tools import create_database_tools, DatabaseTools
from database_manager import DatabaseManager

logger = logging.getLogger(__name__)

# Groq configuration. Keep the model configurable so the application can be
# tested with another Groq-supported model without changing source code.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


class NL2SQLAgents:
    """Factory class for creating specialized NL2SQL agents"""

    def __init__(self, db_manager: DatabaseManager, model_name: str = DEFAULT_GROQ_MODEL):
        self.db_manager = db_manager

        # Keep compatibility with older callers that still pass gpt-4o.
        # The application never sends requests to OpenAI; it maps that legacy
        # value to the configured Groq model instead.
        if model_name == "gpt-4o":
            model_name = DEFAULT_GROQ_MODEL

        self.model_name = model_name
        self.db_tools = create_database_tools(db_manager)

        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not configured. Set your Groq API key in the "
                "environment before starting the application."
            )

        self.llm = ChatGroq(
            model=self.model_name,
            temperature=0.0,
            api_key=GROQ_API_KEY,
            max_tokens=1000,
            timeout=30,
            max_retries=1
        )

    def create_schema_analyst_agent(self) -> Agent:
        """Create Schema Analyst Agent."""
        return Agent(
            role="Database Schema Analyst",
            goal="Analyze and understand database schema to provide comprehensive context for SQL generation",
            backstory="""You are an expert database analyst with deep knowledge of relational database design, 
            normalization principles, and schema optimization. You excel at understanding complex database 
            structures and relationships between tables. Your expertise helps other agents understand 
            the database context needed for accurate SQL generation.""",
            verbose=False,
            allow_delegation=False,
            llm=self.llm,
            system_message=f"""You are a database schema expert. Your primary responsibilities include:

1. **Schema Analysis**: Thoroughly analyze database structure, table relationships, and constraints
2. **Context Provision**: Provide clear, comprehensive schema context for SQL generation
3. **Relationship Mapping**: Identify and explain foreign key relationships and join possibilities
4. **Data Understanding**: Understand data types, constraints, and business logic embedded in schema

**IMPORTANT**: You have access to actual database tools. Use the database manager methods to:
- Call `{self.db_tools.__class__.__name__}.analyze_schema()` to get real table information
- Call `{self.db_tools.__class__.__name__}.describe_table(table_name)` for detailed table info
- Access actual database schema, not hypothetical examples

Key principles:
- Always use the actual database connection to analyze real schema
- Identify primary and foreign key relationships from actual database
- Understand actual data types and constraints
- Provide real sample data context
- Explain actual table purposes and relationships

When analyzing schema:
- Use database tools to get real table names and their purposes
- Get actual column information with proper data types
- Find real relationships between tables
- Use actual constraints and considerations
- Provide real sample data to illustrate table contents

Your analysis should be based on the ACTUAL connected database, not hypothetical examples."""
        )

    def create_sql_generator_agent(self) -> Agent:
        """Create SQL Generator Agent."""
        return Agent(
            role="SQL Query Generator",
            goal="Convert natural language questions into accurate, efficient SQL queries using provided schema context",
            backstory="""You are a senior SQL developer with extensive experience in writing complex queries 
            across different database systems. You have a deep understanding of SQL optimization, query 
            performance, and best practices. You excel at interpreting natural language requirements and 
            translating them into precise SQL statements that leverage the database schema effectively.""",
            verbose=False,
            allow_delegation=False,
            llm=self.llm,
            system_message="""You are an expert SQL query generator. Your responsibilities include:

1. **Query Generation**: Convert natural language questions into accurate SQL queries
2. **Schema Utilization**: Use provided schema context to ensure correct table and column names
3. **Query Optimization**: Write efficient, well-structured SQL queries
4. **Type Awareness**: Understand different query types and apply appropriate SQL patterns

**Query Type Patterns:**
- **Counting**: Use COUNT(), consider DISTINCT, may need GROUP BY
- **Filtering**: Use WHERE clauses, SELECT specific columns
- **Aggregation**: Use AVG(), SUM(), MAX(), MIN(), consider GROUP BY and HAVING
- **Ranking**: Use ORDER BY with LIMIT/TOP, consider DESC for highest values
- **Comparison**: Use comparison operators, multiple conditions
- **Detail**: Select specific columns, use WHERE for specific records

**Best Practices:**
- Always use exact table and column names from schema
- Apply appropriate WHERE clauses for filtering
- Use proper JOIN syntax when multiple tables are needed
- Consider performance implications of queries
- Use proper data type handling
- Round aggregation results to 2 decimal places when appropriate

**SQLite-Specific Syntax:**
- Use strftime() for date operations, NOT EXTRACT()
- For month: strftime('%m', date_column)
- For day: strftime('%d', date_column)
- For year: strftime('%Y', date_column)
- Use double quotes for column names with spaces if needed
- Use LIMIT instead of TOP for limiting results

**Thought Process:**
1. Analyze the natural language question
2. Identify the query type
3. Determine required tables and columns from schema
4. Identify JOIN requirements
5. Apply appropriate WHERE, GROUP BY, ORDER BY, LIMIT clauses
6. Validate query syntax and logic

Always think step-by-step and explain your reasoning before generating the final SQL query."""
        )

    def create_sql_evaluator_agent(self) -> Agent:
        """Create SQL Evaluator Agent."""
        return Agent(
            role="SQL Query Evaluator",
            goal="Validate SQL queries for correctness, execute them safely, and provide detailed feedback on results",
            backstory="""You are a database administrator and quality assurance specialist with extensive 
            experience in SQL validation, query optimization, and database security. You have a keen eye 
            for spotting potential issues in SQL queries and ensuring they execute safely and efficiently. 
            You excel at debugging SQL problems and providing constructive feedback.""",
            verbose=False,
            allow_delegation=False,
            llm=self.llm,
            system_message="""You are a SQL query evaluator and validator. Your responsibilities include:

1. **Query Validation**: Check SQL syntax, table names, column names, and logic
2. **Safety Assessment**: Ensure queries are safe to execute and won't cause issues
3. **Execution**: Execute validated queries and capture results
4. **Performance Evaluation**: Assess query performance and suggest optimizations
5. **Error Handling**: Provide clear error messages and debugging guidance

**Validation Checklist:**
- Syntax correctness
- Table existence and correct names
- Column existence and correct names
- Data type compatibility
- JOIN logic correctness
- WHERE clause validity
- Aggregate function usage
- ORDER BY and LIMIT appropriateness

**Safety Considerations:**
- Avoid queries that could cause excessive resource usage
- Check for potential data exposure issues
- Validate query complexity and execution time

When evaluating queries:
1. First validate syntax and schema references
2. Check for logical correctness
3. Assess performance implications
4. Execute if safe and valid
5. Provide detailed feedback on results
6. Suggest improvements if needed

Always prioritize safety and correctness over speed."""
        )

    def create_result_interpreter_agent(self) -> Agent:
        """Create Result Interpreter Agent."""
        return Agent(
            role="Business Intelligence Analyst",
            goal="Interpret SQL query results and explain them in clear, business-friendly terms",
            backstory="""You are a senior business intelligence analyst with extensive experience in 
            translating technical data insights into actionable business information. You excel at 
            understanding the business context behind data queries and presenting findings in a way 
            that stakeholders can easily understand and act upon.""",
            verbose=False,
            allow_delegation=False,
            llm=self.llm,
            system_message="""You are a business intelligence analyst specializing in identifying database resources used in queries.

Your ONLY responsibility is to identify the specific database tables and columns used by a SQL query.

**Required Output Format:**
**Tables and Columns Used**: [List the specific database tables and columns queried]

Do not provide executive summaries, detailed analysis, or business insights."""
        )

    def get_all_agents(self) -> Dict[str, Agent]:
        """Get all agents as a dictionary."""
        return {
            "schema_analyst": self.create_schema_analyst_agent(),
            "sql_generator": self.create_sql_generator_agent(),
            "sql_evaluator": self.create_sql_evaluator_agent()
        }

    def get_agent_by_name(self, name: str) -> Agent:
        """Get a specific agent by name."""
        agents = self.get_all_agents()
        if name not in agents:
            raise ValueError(f"Agent '{name}' not found. Available agents: {list(agents.keys())}")
        return agents[name]

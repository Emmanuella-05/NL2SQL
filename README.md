# Natural Language to SQL (NL2SQL)

Convert natural language questions into SQL queries using an AI-powered multi-agent system.

## Features

- **Natural Language Processing**: Ask questions in plain English
- **Multi-Database Support**: SQLite, PostgreSQL, MySQL
- **AI-Powered**: Uses Groq models through LangChain + CrewAI
- **Interactive Interface**: Streamlit web application
- **Human Feedback**: Thumbs up/down system for human-in-the-loop learning
- **Ground Truth Testing**: Compare AI queries with your own SQL
- **Sample Dataset**: Pre-loaded NBA database with sample questions for instant testing
- **Performance Optimization**: Fast mode to skip schema analysis for repeat queries
- **Schema Caching**: Reuse analyzed schemas across queries
- **Three-agent architecture**: Schema Analyst, SQL Generator, SQL Evaluator

## LLM provider: Groq

This version uses the Groq API instead of the OpenAI API. The agents use `ChatGroq` through the `langchain-groq` package, so the existing CrewAI architecture can continue to use the same agent/task workflow.

Groq documents `llama-3.3-70b-versatile` as a supported model and provides a free usage tier subject to its current rate and usage limits. See the official Groq documentation for current limits and available models.

## How to run locally

### 1. Clone and install

```bash
git clone https://github.com/Emmanuella-05/NL2SQL.git
cd NL2SQL
pip install -r requirements.txt
```

### 2. Configure Groq

Create a `.env` file based on `.env.example`:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

Do **not** commit your real API key to GitHub.

### 3. Start the application

```bash
streamlit run main.py
```

### 4. Test with the NBA database

Click **Try Sample NBA Dataset**, then try questions such as:

- How many teams are in the NBA?
- List all teams from California
- Who are the players with 'James' in their name?
- How many players are in the database?

## Architecture

The application uses three specialized CrewAI agents:

- **Schema Analyst**: analyzes the real database schema and relationships.
- **SQL Generator**: converts the natural-language question into SQL using the schema context.
- **SQL Evaluator**: validates and evaluates the generated SQL and participates in the retry workflow when needed.

```text
User question
     |
     v
Schema Analyst
     |
     v
SQL Generator
     |
     v
SQL Evaluator
     |
     v
SQL query + database results
```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | Groq API key |
| `GROQ_MODEL` | No | Groq model; defaults to `llama-3.3-70b-versatile` |

## Docker

```bash
docker build -t nl2sql .
docker run -p 8501:8501 -e GROQ_API_KEY=your_key nl2sql
```

## Important note about the free tier

Using Groq does not mean unlimited API usage. The free tier has rate/usage limits that can change. For a test/demo of this NL2SQL project, however, it avoids requiring an OpenAI API key.

## License

MIT License - see LICENSE file for details.

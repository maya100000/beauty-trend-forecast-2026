##Environment Variables##

This project uses the OpenAI API for the LLM + RAG functionality.

Set the API key as an environment variable before running the app.

This project uses the OpenAI API for the LLM + RAG functionality.

Set the API key as an environment variable before running the app.

-Windows (PowerShell)
$env:OPENAI_API_KEY="your_api_key_here"

-macOS / Linux
export OPENAI_API_KEY="your_api_key_here"

⚠️ Never commit your API key to GitHub.
The key is loaded locally via environment variables only.

If the API key is not set, the application will still run,
but the “Ask the Project” (LLM) page will be disabled.
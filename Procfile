web: uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
dashboard: streamlit run dashboard/app.py --server.port ${PORT:-8501} --server.address 0.0.0.0
agent: streamlit run dashboard/agent.py --server.port ${PORT:-8502} --server.address 0.0.0.0
worker: python jobs/forecast_poll.py

# The dashboard, packaged. Build: docker build -t nothing-but-pixels .
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dqn_catch.py app.py ./

EXPOSE 8501

# headless: don't ask for an email or try to open a browser inside the container.
# address 0.0.0.0: listen on the container's network, not just its loopback.
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.headless=true"]

FROM python:3.11.14
WORKDIR /app

# Install the application dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy in the source code
COPY api.py utils.py ./
COPY static/ ./static/
EXPOSE 8080

ENV LANGCHAIN_TRACING_V2=true

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
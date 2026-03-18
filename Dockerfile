FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser
EXPOSE 8080
CMD [ "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1" ]
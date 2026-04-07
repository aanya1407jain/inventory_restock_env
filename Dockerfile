FROM python:3.11

# Create a user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy requirements from the root (make sure you uploaded it to the main folder!)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy everything else
COPY --chown=user . .

# IMPORTANT: This tells Python to look in the 'server' folder for imports
ENV PYTHONPATH=/app:/app/server

# Run the app using the 'server' folder path
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]

FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY . .

# Install dependencies using uv
RUN uv sync

# Expose the port
EXPOSE 1416

# Run the application
CMD ["uv", "run", "python", "app.py"]
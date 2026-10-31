FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY tests ./tests
COPY configs ./configs
COPY examples ./examples
RUN pip install --no-cache-dir -e ".[dev]"
CMD ["qalpha", "doctor"]

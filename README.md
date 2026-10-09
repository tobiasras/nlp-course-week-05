# NLP course week 05

FastAPI backend on port **8000**.

Add jsonl data to the data/ <-- here


## Run

```bash
docker compose up --build
```

- API docs: http://127.0.0.1:8000/docs

```bash
curl -s http://localhost:8000/
```

## Tests

```bash
python -m pytest test/backend/test_main.py
```

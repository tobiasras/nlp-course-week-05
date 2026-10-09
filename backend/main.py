import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException

from course_service import CourseSearchService

service = CourseSearchService()

def courses_path() -> Path:
    """Return the path to the course JSONL file."""
    configured = os.environ.get("COURSES_PATH")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[1] / "data" / "dtu_courses.jsonl"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the course index when the app starts."""
    service.init(courses_path())
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    """Check that the app is running."""
    return {"message": "app is running"}


@app.get("/v1/courses/{course_id}/similar")
async def similar_courses(course_id: str, top_k: int = 10):
    """Return courses closest to the given course."""
    try:
        results = service.similar(course_id, top_k=top_k)
    except KeyError:
        raise HTTPException(status_code=404, detail="course not found") from None
    result = {
        "query_course_id": course_id,
        "results": results,
        "top_k": top_k,
    }
    print(result)
    return result


@app.get("/v1/search")
async def search_courses(query: str, top_k: int = 10):
    """Search courses by free text."""
    result = {
        "query": query,
        "results": service.search(query, top_k=top_k),
        "top_k": top_k,
    }
    print(result)
    return result


@app.get("/v1/objectives/search")
async def search_objectives(query: str, top_k: int = 10):
    """Search learning objectives by free text."""
    result = {
        "query": query,
        "results": service.search_objectives(query, top_k=top_k),
        "top_k": top_k,
    }
    print(result)
    return result


@app.get("/v1/health")
async def health():
    """Report index sizes."""
    result = service.health()
    print(result)
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

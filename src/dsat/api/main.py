"""FastAPI application for DSAT.

Exposes Kedro pipelines as REST API endpoints.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dsat.api.routes import eda, feature_engineering, dag, leakage

app = FastAPI(
    title="Data Science Assistant Tool (DSAT)",
    description="Kedro-based Data Science Assistant with FastAPI endpoints",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers with same prefixes as original
app.include_router(eda.router, prefix="/EDA", tags=["EDA"])
app.include_router(feature_engineering.router, prefix="/FeatureEngineering", tags=["Feature Engineering"])
app.include_router(dag.router, prefix="/Transformation", tags=["DAG Generation"])
app.include_router(leakage.router, prefix="/LeakageDetection", tags=["Leakage Detection"])


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "DSAT", "framework": "Kedro + FastAPI"}


@app.get("/health")
async def health():
    """Health check for load balancers."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

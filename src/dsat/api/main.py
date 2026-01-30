"""FastAPI application for DSAT.

Exposes Kedro pipelines as REST API endpoints.
Matches original DataScienceAssistantTool endpoints.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dsat.api.routes import eda, feature_engineering, dag, leakage, balance

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

# Register routers with same prefixes as original application
app.include_router(eda.router, prefix="/EDA", tags=["EDA"])
app.include_router(feature_engineering.router, prefix="/Feature Engineering", tags=["Feature Engineering"])
app.include_router(dag.router, prefix="/Transformation", tags=["Transformation"])
app.include_router(leakage.router, prefix="/Leakage Detection", tags=["Leakage Detection"])
app.include_router(balance.router, prefix="/DataBalancing", tags=["Data Balancing"])


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

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse,HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
import json
import time
import uuid
from .database import SessionLocal, engine, Base
from .models import RequestHistory
from sqlalchemy.orm import Session
from sklearn.cluster import KMeans
from sqlalchemy import desc
from pathlib import Path
import os
Base.metadata.create_all(bind=engine)
class PortfolioPipeline:
    def __init__(self, decay_factor=0.95, n_clusters=8, n_iterations=10):
        self.decay_factor = decay_factor
        self.n_clusters = n_clusters
        self.n_iterations = n_iterations
        self.is_trained = False
        
    def fit(self, tensor_3d, tickers):
        self.tickers_ = tickers
        n_days, n_tickers, n_features = tensor_3d.shape
        
        self.distance_matrix_ = np.zeros((n_tickers, n_tickers))
        for i in range(n_tickers):
            for j in range(i+1, n_tickers):
                diff = np.abs(tensor_3d[:, i, :] - tensor_3d[:, j, :])
                weights = self.decay_factor ** np.arange(n_days)
                dist = np.sum(diff * weights[:, np.newaxis])
                self.distance_matrix_[i, j] = dist
                self.distance_matrix_[j, i] = dist
        
        similarity = np.max(self.distance_matrix_) - self.distance_matrix_
        best_inertia = float('inf')
        
        for i in range(self.n_iterations):
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42+i, n_init=1)
            labels = kmeans.fit_predict(similarity)
            if kmeans.inertia_ < best_inertia:
                best_inertia = kmeans.inertia_
                self.labels_ = labels
                self.kmeans_ = kmeans
        
        self.is_trained = True
        return self
    
    def predict(self):
        if not self.is_trained:
            raise ValueError("Модель не обучена")
        
        portfolio = []
        for cluster_id in range(self.n_clusters):
            indices = np.where(self.labels_ == cluster_id)[0]
            if len(indices) > 0:
                portfolio.append(self.tickers_[indices[0]])
        return portfolio
    
    def get_clusters(self):
        if not self.is_trained:
            raise ValueError("Модель не обучена")
        
        clusters = {}
        for cluster_id in range(self.n_clusters):
            indices = np.where(self.labels_ == cluster_id)[0]
            if len(indices) > 0:
                clusters[f"cluster_{cluster_id}"] = {  # <-- Более читаемый ключ
                    'tickers': [self.tickers_[i] for i in indices],
                    'size': len(indices),
                    'center': self.tickers_[indices[0]]
                }
        return clusters



def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
app = FastAPI(title="Portfolio ML Service")

global_pipeline = None

class PortfolioRequest(BaseModel):
    tensor_3d: List[List[List[float]]]
    tickers: List[str] 

class PredictionResult(BaseModel):
    portfolio: List[str]
    clusters: Dict[str, Any] 
    processing_time: float
    n_days: int
    n_tickers: int
    n_features: int
    request_id: str

class HistoryResponse(BaseModel):
    id: int
    endpoint: str
    method: str
    processing_time: float
    status_code: int
    created_at: str
    input_preview: Dict[str, Any]
    output_preview: Dict[str, Any]

@app.get("/")
async def root():
 
    return {
        "service": "Portfolio ML Service",
        "status": "running",
        "model_ready": global_pipeline is not None and global_pipeline.is_trained
    }
@app.post("/forward", response_model=PredictionResult)
async def forward(request: PortfolioRequest, db: Session = Depends(get_db)):
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]
    
    try:
        tensor_np = np.array(request.tensor_3d, dtype=np.float64)
        tickers_np = np.array(request.tickers)
        
        if len(tensor_np.shape) != 3:
            raise HTTPException(status_code=400, detail="Тензор должен быть 3D")
        
        n_days, n_tickers, n_features = tensor_np.shape
        
        if len(tickers_np) != n_tickers:
            raise HTTPException(
                status_code=400,
                detail=f"Количество тикеров не совпадает"
            )
        
        pipeline = PortfolioPipeline(
            decay_factor=0.95,
            n_clusters=min(8, n_tickers),
            n_iterations=10
        )
        
        pipeline.fit(tensor_np, tickers_np)
        
        portfolio = pipeline.predict()
        clusters = pipeline.get_clusters()
        
        processing_time = time.time() - start_time
        
        result = {
            "portfolio": portfolio,
            "clusters": clusters,
            "processing_time": round(processing_time, 4),
            "n_days": n_days,
            "n_tickers": n_tickers,
            "n_features": n_features,
            "request_id": request_id
        }
        
        history_record = RequestHistory(
            endpoint="/forward",
            method="POST",
            input_data={
                "tensor_shape": [n_days, n_tickers, n_features],
                "tickers": tickers_np.tolist(),
                "tickers_count": n_tickers
            },
            output_data=result,
            processing_time=processing_time,
            status_code=200,
            error_message=None
        )
        
        db.add(history_record)
        db.commit()
        db.refresh(history_record)
        
        return result
        
    except HTTPException as he:
        error_record = RequestHistory(
            endpoint="/forward",
            method="POST",
            input_data={"tensor_shape": [], "tickers": request.tickers},
            output_data=None,
            processing_time=time.time() - start_time,
            status_code=he.status_code,
            error_message=str(he.detail)
        )
        
        db.add(error_record)
        db.commit()
        raise he
        
    except Exception as e:
        error_record = RequestHistory(
            endpoint="/forward",
            method="POST",
            input_data={"error": "invalid_input"},
            output_data=None,
            processing_time=time.time() - start_time,
            status_code=500,
            error_message=str(e)
        )
        
        db.add(error_record)
        db.commit()
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")

@app.get("/history", response_model=List[HistoryResponse])
async def get_history(
    skip: int = 0,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    history = db.query(RequestHistory) \
        .order_by(RequestHistory.created_at.desc()) \
        .offset(skip) \
        .limit(limit) \
        .all()
    
    response = []
    for record in history:
        response.append({
            "id": record.id,
            "endpoint": record.endpoint,
            "method": record.method,
            "processing_time": record.processing_time,
            "status_code": record.status_code,
            "created_at": record.created_at.isoformat(),
            "input_preview": {
                "tickers_count": record.input_data.get("tickers_count", 0) if record.input_data else 0,
                "tensor_shape": record.input_data.get("tensor_shape", []) if record.input_data else [],
                "has_tickers": "tickers" in record.input_data if record.input_data else False
            },
            "output_preview": {
                "has_portfolio": "portfolio" in record.output_data if record.output_data else False,
                "portfolio_count": len(record.output_data.get("portfolio", [])) if record.output_data else 0,
                "clusters_count": len(record.output_data.get("clusters", {})) if record.output_data else 0
            }
        })
    
    return response

@app.get("/eda1", response_class=HTMLResponse)
async def get_html_file():
    current_dir = Path(__file__).parent
    html_path = current_dir.parent / "EDA.html"
  
    return FileResponse(html_path, media_type="text/html")
@app.get("/eda2", response_class=HTMLResponse)
async def get_html_file():
    current_dir = Path(__file__).parent
    html_path = current_dir.parent / "EDAnew.html"

    return FileResponse(html_path, media_type="text/html")

@app.get("/pres")
async def get_presentation():
    current_dir = Path(__file__).parent
    pptx_path = current_dir.parent / "Презентация.pptx"
    
    return FileResponse(
        pptx_path,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename="presentation.pptx"
    )
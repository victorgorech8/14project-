from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
import time
import uuid
from .database import SessionLocal, engine, Base
from .finam_trade import get_finam_account_info, get_finam_status, place_finam_orders
from .models import RequestHistory
from .moex_dynamic import (
    build_dynamic_portfolio,
    build_rebalance_plan,
    get_dynamic_portfolio_weights,
    get_top_stocks,
)
from .portfolio_advanced import AdvancedPortfolioPipeline
from .softs_forecast import build_softs_signal, get_softs_analysis, tsururu_example_code
from .ui import render_dashboard
from sqlalchemy.orm import Session
from pathlib import Path
Base.metadata.create_all(bind=engine)
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
    method: str = "ensemble"
    target_assets: Optional[int] = None
    max_weight: float = 0.25

class DynamicPortfolioRequest(BaseModel):
    lookback_days: int = 180
    top_pool: int = 30
    target_assets: int = 8
    method: str = "ensemble"
    max_weight: float = 0.25
    universe_limit: int = 60
    cache_ttl_seconds: int = 1800

class RebalancePlanRequest(DynamicPortfolioRequest):
    cash: float = 100000.0
    current_positions: Dict[str, int] = {}

class FinamOrderRequest(BaseModel):
    ticker: str
    side: str = "BUY"
    quantity: int
    price: Optional[float] = None

class FinamOrdersRequest(BaseModel):
    confirm_real_trading: bool = False
    orders: List[FinamOrderRequest]

class PredictionResult(BaseModel):
    portfolio: List[str]
    weights: Dict[str, float]
    clusters: Dict[str, Any] 
    diagnostics: Dict[str, Any]
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
    return render_dashboard()

@app.get("/api")
async def api_root():
    return {
        "service": "Portfolio ML Service",
        "status": "running",
        "model_ready": global_pipeline is not None and global_pipeline.is_trained,
        "portfolio_methods": AdvancedPortfolioPipeline.METHODS
    }

@app.get("/methods")
async def get_methods():
    return {
        "methods": AdvancedPortfolioPipeline.METHODS,
        "default": "ensemble",
        "input_note": "tensor_3d должен иметь форму [days, tickers, features]; первая feature используется как цена."
    }

@app.get("/finam/status")
async def finam_status():
    return get_finam_status()

@app.get("/finam/account")
async def finam_account():
    try:
        return await get_finam_account_info()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ошибка Finam API: {str(e)}")

@app.post("/finam/orders")
async def finam_orders(request: FinamOrdersRequest):
    try:
        return await place_finam_orders(
            orders=[order.model_dump() for order in request.orders],
            confirm_real_trading=request.confirm_real_trading,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ошибка отправки заявок Finam: {str(e)}")

@app.get("/research/softs")
async def research_softs():
    return get_softs_analysis()

@app.get("/research/softs-signal")
async def research_softs_signal(
    lookback_days: int = 180,
    top_pool: int = 30,
    horizon_days: int = 5,
    universe_limit: int = 60,
):
    try:
        return build_softs_signal(
            lookback_days=lookback_days,
            top_pool=top_pool,
            horizon_days=horizon_days,
            universe_limit=universe_limit,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ошибка SOFTS-сигнала: {str(e)}")

@app.get("/research/tsururu-example")
async def research_tsururu_example():
    return tsururu_example_code()

@app.get("/moex/top-stocks")
async def moex_top_stocks(
    lookback_days: int = 90,
    limit: int = 10,
    universe_limit: int = 40,
    cache_ttl_seconds: int = 1800,
):
    try:
        return get_top_stocks(
            lookback_days=lookback_days,
            limit=limit,
            universe_limit=universe_limit,
            cache_ttl_seconds=cache_ttl_seconds,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ошибка получения данных MOEX: {str(e)}")

@app.post("/moex/dynamic-portfolio")
async def moex_dynamic_portfolio(request: DynamicPortfolioRequest):
    try:
        return build_dynamic_portfolio(
            lookback_days=request.lookback_days,
            top_pool=request.top_pool,
            target_assets=request.target_assets,
            method=request.method,
            max_weight=request.max_weight,
            universe_limit=request.universe_limit,
            cache_ttl_seconds=request.cache_ttl_seconds,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ошибка динамического портфеля MOEX: {str(e)}")

@app.get("/moex/dynamic-portfolio")
async def moex_dynamic_portfolio_get(
    lookback_days: int = 180,
    top_pool: int = 30,
    target_assets: int = 8,
    method: str = "ensemble",
    max_weight: float = 0.25,
    universe_limit: int = 60,
    cache_ttl_seconds: int = 1800,
):
    try:
        return build_dynamic_portfolio(
            lookback_days=lookback_days,
            top_pool=top_pool,
            target_assets=target_assets,
            method=method,
            max_weight=max_weight,
            universe_limit=universe_limit,
            cache_ttl_seconds=cache_ttl_seconds,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ошибка динамического портфеля MOEX: {str(e)}")

@app.get("/moex/dynamic-portfolio/weights")
async def moex_dynamic_portfolio_weights(
    lookback_days: int = 180,
    top_pool: int = 30,
    target_assets: int = 8,
    method: str = "ensemble",
    max_weight: float = 0.25,
    universe_limit: int = 60,
    cache_ttl_seconds: int = 1800,
):
    try:
        return get_dynamic_portfolio_weights(
            lookback_days=lookback_days,
            top_pool=top_pool,
            target_assets=target_assets,
            method=method,
            max_weight=max_weight,
            universe_limit=universe_limit,
            cache_ttl_seconds=cache_ttl_seconds,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ошибка весов динамического портфеля MOEX: {str(e)}")

@app.post("/moex/rebalance-plan")
async def moex_rebalance_plan(request: RebalancePlanRequest):
    try:
        return build_rebalance_plan(
            cash=request.cash,
            current_positions=request.current_positions,
            lookback_days=request.lookback_days,
            top_pool=request.top_pool,
            target_assets=request.target_assets,
            method=request.method,
            max_weight=request.max_weight,
            universe_limit=request.universe_limit,
            cache_ttl_seconds=request.cache_ttl_seconds,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ошибка построения плана ребалансировки: {str(e)}")

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
        request_method = request.method.lower()
        if request_method not in AdvancedPortfolioPipeline.METHODS:
            raise HTTPException(
                status_code=400,
                detail=f"Неизвестный метод '{request.method}'. Доступно: {list(AdvancedPortfolioPipeline.METHODS)}"
            )
        if request.target_assets is not None and request.target_assets <= 0:
            raise HTTPException(status_code=400, detail="target_assets должен быть положительным")
        if not 0 < request.max_weight <= 1:
            raise HTTPException(status_code=400, detail="max_weight должен быть в диапазоне (0, 1]")
         
        pipeline = AdvancedPortfolioPipeline(
            decay_factor=0.97,
            n_clusters=min(8, n_tickers),
            n_iterations=10,
            method=request_method,
            target_assets=request.target_assets,
            max_weight=request.max_weight
        )
         
        pipeline.fit(tensor_np, tickers_np)
         
        portfolio = pipeline.predict()
        weights = pipeline.get_weights()
        clusters = pipeline.get_clusters()
        diagnostics = pipeline.get_diagnostics()
         
        processing_time = time.time() - start_time
         
        result = {
            "portfolio": portfolio,
            "weights": weights,
            "clusters": clusters,
            "diagnostics": diagnostics,
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
                "tickers_count": n_tickers,
                "method": request_method,
                "target_assets": request.target_assets,
                "max_weight": request.max_weight
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
                "method": record.input_data.get("method", "legacy") if record.input_data else "legacy",
                "has_tickers": "tickers" in record.input_data if record.input_data else False
            },
            "output_preview": {
                "has_portfolio": "portfolio" in record.output_data if record.output_data else False,
                "portfolio_count": len(record.output_data.get("portfolio", [])) if record.output_data else 0,
                "has_weights": "weights" in record.output_data if record.output_data else False,
                "clusters_count": len(record.output_data.get("clusters", {})) if record.output_data else 0
            }
        })
    
    return response

@app.get("/eda1", response_class=HTMLResponse)
async def get_html_file():
    current_dir = Path(__file__).parent
    html_path = current_dir.parent / "analysis" / "reports" / "EDA.html"
  
    return FileResponse(html_path, media_type="text/html")
@app.get("/eda2", response_class=HTMLResponse)
async def get_html_file():
    current_dir = Path(__file__).parent
    html_path = current_dir.parent / "analysis" / "reports" / "EDAnew.html"

    return FileResponse(html_path, media_type="text/html")

@app.get("/pres")
async def get_presentation():
    current_dir = Path(__file__).parent
    pptx_path = current_dir.parent / "docs" / "Презентация.pptx"
    
    return FileResponse(
        pptx_path,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename="presentation.pptx"
    )

@app.get("/methodology")
async def get_methodology_pdf():
    current_dir = Path(__file__).parent
    pdf_path = current_dir.parent / "docs" / "portfolio_methodology.pdf"

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename="portfolio_methodology.pdf"
    )

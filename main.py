from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uuid
from typing import Optional, Dict, Any
import asyncio
import logging

# Импортируем твой pipeline (создадим дальше)
from pipeline import analyze_site

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создаём приложение
app = FastAPI(
    title="Design Analyzer API",
    description="API для анализа дизайна веб-сайтов",
    version="1.0.0"
)

# Разрешаем запросы с фронтенда (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшне замени на домен твоего фронтенда
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Модели данных
class AnalyzeRequest(BaseModel):
    url: str
    callback_url: Optional[str] = None  # если хочешь уведомление

class AnalyzeResponse(BaseModel):
    task_id: str
    status: str
    message: str

class TaskStatus(BaseModel):
    task_id: str
    status: str  # "processing", "completed", "failed"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Хранилище задач (в продакшне замени на Redis/БД)
tasks_db = {}

# Эндпоинт для запуска анализа
@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    Запускает анализ сайта по URL.
    Возвращает task_id для отслеживания статуса.
    """
    task_id = str(uuid.uuid4())
    
    # Сохраняем задачу
    tasks_db[task_id] = {
        "status": "processing",
        "url": request.url,
        "result": None
    }
    
    # Запускаем в фоне (чтобы не блокировать ответ)
    background_tasks.add_task(run_analysis, task_id, request.url)
    
    return AnalyzeResponse(
        task_id=task_id,
        status="processing",
        message="Анализ запущен. Используйте /api/status/{task_id} для проверки"
    )

# Фоновая задача
async def run_analysis(task_id: str, url: str):
    try:
        logger.info(f"Запуск анализа для {url}, task_id: {task_id}")
        
        # Вызываем твой pipeline (из pipeline.py)
        result = await analyze_site(url)
        
        # Сохраняем результат
        tasks_db[task_id]["status"] = "completed"
        tasks_db[task_id]["result"] = result
        
        logger.info(f"Анализ завершён для {task_id}")
        
    except Exception as e:
        logger.error(f"Ошибка анализа {task_id}: {str(e)}")
        tasks_db[task_id]["status"] = "failed"
        tasks_db[task_id]["error"] = str(e)

# Эндпоинт для проверки статуса
@app.get("/api/status/{task_id}", response_model=TaskStatus)
async def get_status(task_id: str):
    """
    Возвращает статус задачи по task_id.
    """
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks_db[task_id]
    return TaskStatus(
        task_id=task_id,
        status=task["status"],
        result=task.get("result"),
        error=task.get("error")
    )

# Эндпоинт для проверки здоровья сервера
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": str(asyncio.get_event_loop().time())
    }

# Корневой эндпоинт
@app.get("/")
async def root():
    return {
        "name": "Design Analyzer API",
        "version": "1.0.0",
        "endpoints": {
            "POST /api/analyze": "Запустить анализ",
            "GET /api/status/{task_id}": "Проверить статус",
            "GET /health": "Проверка здоровья",
            "GET /docs": "Документация Swagger"
        }
    }

# Запуск (для локальной разработки)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
import uvicorn
from dotenv import load_dotenv

load_dotenv()  # загружает переменные из .env

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,   # автоперезагрузка при изменении файлов
    )

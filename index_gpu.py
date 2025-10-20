
import uvicorn


if __name__ == '__main__':    
    uvicorn.run("main_gpu:app",host="0.0.0.0",port=59532,reload=False)
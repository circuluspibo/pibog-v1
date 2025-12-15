import uvicorn

if __name__ == '__main__':    
    uvicorn.run("main_cpu_sr:app",host="0.0.0.0",port=59530,reload=False)

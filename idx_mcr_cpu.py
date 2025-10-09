import uvicorn

if __name__ == '__main__':    
    uvicorn.run("main_cpu_mcr:app",host="0.0.0.0",port=8000,reload=False)

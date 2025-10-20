import uvicorn

if __name__ == '__main__':    
    uvicorn.run("main_npu_mcr:app",host="0.0.0.0",port=59531,reload=False)

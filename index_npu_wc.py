
import uvicorn

if __name__ == '__main__':    
    uvicorn.run("main_npu_wc:app",host="0.0.0.0",port=59533,reload=False)

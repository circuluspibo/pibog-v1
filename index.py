
import uvicorn

_PORT = 59531

f = open("port.txt", 'w')
f.write(str(_PORT))
f.close()

if __name__ == '__main__':    
    uvicorn.run("main:app",host="0.0.0.0",port=_PORT,reload=False)

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Pension Planner API is running"}

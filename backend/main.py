from fastapi import FastAPI
from pydantic import BaseModel
from backend import run_optimization

app=FastAPI()

class Input(BaseModel):
    user_input: str
    
@app.post('/optimize')
def optimize(input:Input):
    enhanced_prompt = run_optimization(input.user_input)
    return {'enhanced_prompt': enhanced_prompt}
    
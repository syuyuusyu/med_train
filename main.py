from app import create_app
import os
os.environ["PYTHONPYCACHEPREFIX"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "cache"))

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)
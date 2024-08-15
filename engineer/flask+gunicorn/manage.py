from flask import Flask, g
import threading
import os

def create_app():
    app = Flask(__name__)
    
    # Registering the blueprint for predictions
    from service import predict_bp  # Adjust the import according to your project structure
    app.register_blueprint(predict_bp, url_prefix='/predict')

    # Function to download or load the model asynchronously
    from service.predict import G_model
    def download_model():
        try:
            load_model = threading.Thread(target=G_model.async_load_model)
            load_model.start()
            load_model.join()  # Ensure the model is loaded before proceeding
        except Exception as e:
            print("Error loading model asynchronously:", e)

    @app.before_request
    def before_request_func():
        if not os.path.exists('/root/autodl-tmp/model/qwen'):
            download_model()

        if not G_model.llm:
            G_model.load_model()
            
    return app

app = create_app()

if __name__ == "__main__":
    app.run(host = '127.0.0.1', port='6006')
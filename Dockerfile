FROM python:3.12-slim
WORKDIR /app
COPY requirements-api.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt
COPY . .
# Treina e gera o modelo (dados sintéticos por padrão); depois sobe a API.
RUN python main.py --no_explain --n_bootstrap 50
ENV MODEL_PATH=outputs/modelos/modelo_fraude.joblib
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

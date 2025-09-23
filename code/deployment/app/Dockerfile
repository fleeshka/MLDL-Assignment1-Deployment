FROM python:3.10-slim

WORKDIR /app

COPY ./deployment/app/requirements.txt ./requirements.txt


RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


COPY . .

CMD ["streamlit", "run", "deployment/app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

FROM python:3.10.8

COPY . /app
WORKDIR /app

RUN pip install -r /app/requirements.txt

ENTRYPOINT ["streamlit", "run", "app.py"]
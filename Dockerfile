FROM python:3.11.0

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

EXPOSE 8003

CMD ["python", "app.py"]
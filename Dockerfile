FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt dataset.csv ./
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]
FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8099

CMD ["python", "Life Cookbook fixed 试做.py"]
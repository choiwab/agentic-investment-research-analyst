FROM apache/airflow:2.7.0-python3.9


USER airflow
RUN pip3 install --no-cache-dir --user finnhub-python pymongo python-dotenv


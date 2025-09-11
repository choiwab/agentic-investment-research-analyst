FROM apache/airflow:2.7.0-python3.9

USER root
RUN pip3 install --no-cache-dir finnhub-python pymongo
USER airflow


FROM python:3.8-slim-buster

RUN pip install numpy==1.21.0 pandas==1.3.0 scikit-learn==0.24.2 joblib==1.0.1 boto3==1.18.0

COPY /src /opt/ml/code

WORKDIR /opt/ml/code

ENTRYPOINT ["python", "predict.py"]
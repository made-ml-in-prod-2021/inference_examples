FROM python:3.6
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY batch_predict.py /batch_predict.py

WORKDIR .

ENV PATH_TO_MODEL="mlflow/3/10c8f12869bc4b02864773b51a2730d8/artifacts/model.pkl"
ENV OUTPUT="predicts/output.csv"
ENV PATH_TO_DATA="test.csv"

CMD ["python", "batch_predict.py"]
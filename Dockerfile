FROM python:3.10-slim

ARG RUN_ID

CMD echo "Downloading model for Run ID: $RUN_ID"
FROM nvcr.io/nvidia/pytorch:23.05-py3

LABEL title="fa_for_ml" \
      version="1.0" \
      description="An Introduction to Functional Analysis for Machine Learning"

RUN apt-get update && apt-get upgrade -y

WORKDIR /workspace/src
RUN pip install matplotlib numpy cvxpy Gpy gpytorch

FROM ubuntu:latest

RUN apt-get update 
RUN apt-get install -y python3 pip

RUN pip install pandas numpy sklearn xgboost optbinning pandas-profiling

COPY ./ .
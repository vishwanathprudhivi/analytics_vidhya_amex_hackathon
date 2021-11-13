FROM codait/max-object-detector:arm-arm32v7-latest

RUN apt-get update 
RUN apt-get install -y python3

RUN pip install cmake 
RUN pip install tensorflow keras 
RUN pip install pandas sklearn xgboost pandas-profiling 

#RUN pip install keras
COPY ./ .

ENTRYPOINT [ "python3" ]
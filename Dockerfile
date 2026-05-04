FROM python:3.12-slim

LABEL maintainer="te.pickering@gmail.com"

COPY . /src
WORKDIR /src

RUN python -m pip install --upgrade pip && pip install .

COPY scripts/iers.py /usr/local/bin/iers.py
RUN /usr/local/bin/iers.py

VOLUME ["/data"]
WORKDIR /data

ENTRYPOINT ["process_stellacam_dir"]

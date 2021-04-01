FROM python:3.8

LABEL maintainer="te.pickering@gmail.com"

COPY . .

RUN python -m pip install --upgrade pip
RUN pip install -e .[all]

COPY scripts/iers.py /usr/local/bin/iers.py

VOLUME ["/data"]

WORKDIR /data

RUN /usr/local/bin/iers.py

ENTRYPOINT ["process_stellacam_dir"]

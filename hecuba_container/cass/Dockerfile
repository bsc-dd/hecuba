FROM cassandra:4.0
RUN apt-get update && apt-get install -y \
        python3-pip \
  && rm -rf /var/lib/apt/lists/*

RUN pip3 install hecuba

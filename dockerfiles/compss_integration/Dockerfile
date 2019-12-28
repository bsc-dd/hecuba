FROM compss/compss:latest
RUN apt-get update -y && apt-get install -y cmake
RUN pip3 install hecuba
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

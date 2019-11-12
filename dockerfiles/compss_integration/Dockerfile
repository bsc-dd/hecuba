FROM compss/compss:latest
RUN pip3 install hecuba
RUN echo "deb http://www.apache.org/dist/cassandra/debian 311x main" | sudo tee -a /etc/apt/sources.list.d/cassandra.sources.list
RUN curl https://www.apache.org/dist/cassandra/KEYS | sudo apt-key add -
RUN sudo apt-get update \
&& sudo apt-get install cassandra -y
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

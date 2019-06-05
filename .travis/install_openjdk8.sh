# Because of https://travis-ci.community/t/oracle-jdk-11-and-10-are-pre-installed-not-the-openjdk-builds/785/13
# Travis ignores the JDK option and we need to manually install openjdk8
# following https://docs.datastax.com/en/cassandra/3.0/cassandra/install/installOpenJdkDeb.html


sudo add-apt-repository -y ppa:openjdk-r/ppa
sudo apt-get update
sudo apt-get install -y openjdk-8-jdk --no-install-recommends
sudo update-java-alternatives --jre-headless --jre --set java-1.8.0-openjdk-amd64
export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64
java -version

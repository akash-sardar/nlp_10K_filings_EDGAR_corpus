# Use Python 3.10 as base image with Debian Bullseye
FROM python:3.10-bullseye

# Expose ports for Jupyter Lab (8888) and Spark Web UI (4040)
EXPOSE 8888 4040

# Set bash as default shell for better scripting support
SHELL ["/bin/bash", "-c"]

# Update pip to latest version for package management
RUN pip install --upgrade pip

# Update package list and install Java 11 JDK (required for Spark)
RUN apt-get update && \
    apt install -y openjdk-11-jdk && \
    apt-get clean;
    
# Install Java certificates and update certificate store
RUN apt-get install ca-certificates-java && \
    apt-get clean && \
    update-ca-certificates -f;

# Install text editors for container debugging and file editing
RUN apt-get install -y nano && \
    apt-get install -y vim;

# Set JAVA_HOME environment variable for Spark to find Java
ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64/
RUN export JAVA_HOME

# Change to temp directory for downloading Spark
WORKDIR /tmp
# Download Apache Spark 3.5.1 with Hadoop 3 binaries
RUN wget https://archive.apache.org/dist/spark/spark-3.5.1/spark-3.5.1-bin-hadoop3.tgz
# Extract the downloaded tar.gz file
RUN tar -xvf spark-3.5.1-bin-hadoop3.tgz
# Rename extracted directory to 'spark'
RUN mv spark-3.5.1-bin-hadoop3 spark
# Move Spark to root directory for global access
RUN mv spark /
# Clean up downloaded archive file
RUN rm spark-3.5.1-bin-hadoop3.tgz

# Set Spark home directory environment variable
ENV SPARK_HOME /spark
RUN export SPARK_HOME
# Set Python executable for PySpark driver
ENV PYSPARK_PYTHON /usr/local/bin/python
RUN export PYSPARK_PYTHON
# Set Python path to include Spark Python libraries and py4j
ENV PYTHONPATH $SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9.7-src.zip
RUN export PYTHONPATH
# Add Spark binaries to system PATH
ENV PATH $PATH:$SPARK_HOME/bin
RUN export PATH

# Copy Spark configuration templates to active config files
RUN mv $SPARK_HOME/conf/log4j2.properties.template $SPARK_HOME/conf/log4j2.properties
RUN mv $SPARK_HOME/conf/spark-defaults.conf.template $SPARK_HOME/conf/spark-defaults.conf
RUN mv $SPARK_HOME/conf/spark-env.sh.template $SPARK_HOME/conf/spark-env.sh

# Install Python packages: Jupyter, PySpark, Kafka, Delta Lake, AWS tools
RUN pip install --no-cache-dir \
    jupyterlab \
    pyspark==3.5.1 \
    kafka-python \
    delta-spark==3.1.0 \
    boto3 \
    awscli

# Set workspace directory that will be mounted from host
WORKDIR /workspace

# Create data subdirectory for application output files
RUN mkdir -p /workspace/data

# Create IPython profile for Jupyter configuration
RUN ipython profile create
# Fix Jupyter kernel logging issue by disabling file descriptor capture
RUN echo "c.IPKernelApp.capture_fd_output = False" >> "/root/.ipython/profile_default/ipython_kernel_config.py"

# Start Jupyter Lab server accessible from any IP with root privileges
CMD ["python3", "-m", "jupyterlab", "--ip", "0.0.0.0", "--allow-root"]
#!/bin/bash

export CONTACT_NAMES='localhost'
export NODE_PORT=9042

javac StorageItf.java
java StorageItf

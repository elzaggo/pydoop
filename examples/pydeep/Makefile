# BEGIN_COPYRIGHT
# 
# Copyright 2009-2018 CRS4.
# 
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at
# 
#   http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
# 
# END_COPYRIGHT

USER ?= $(shell whoami)
PYTHON ?= python
NUM_MAPS ?= 2
NUM_REDUCERS ?= 2
LOGLEVEL ?= INFO
BNECK_INPUT ?= bneck_input
BNECK_OUTPUT ?= bneck_output

pathsearch = $(firstword $(wildcard $(addsuffix /$(1),$(subst :, ,$(PATH)))))

HDFS=$(if $(call pathsearch,hdfs),$(call pathsearch,hdfs) dfs ,\
       $(if $(call pathsearch,hadoop),$(call pathsearch,hadoop) fs ,\
	       HDFS_IS_MISSING))
HDFS_RMR=$(if $(call pathsearch,hdfs),$(call pathsearch,hdfs) dfs -rm -r,\
	       $(if $(call pathsearch,hadoop),$(call pathsearch,hadoop) fs -rm -r,\
	       HDFS_IS_MISSING))
HDFS_PUT=${HDFS} -put
HDFS_MKDIR=${HDFS} -mkdir


.PHONY: clean distclean dfsclean


data:
	-${HDFS_MKDIR}  /user || :
	-${HDFS_MKDIR} /user/${USER} || :
	-${HDFS_MKDIR} /user/${USER}/${BNECK_INPUT} || :
	-${HDFS_RMR} /user/${USER}/${BNECK_OUTPUT} || :

gen

genbnecks: data
	${PYTHON} genrecords.py --log-level ${LOGLEVEL}\
       --num-maps ${NUM_MAPS}\
       /user/${USER}/${BNECK_INPUT} /user/${USER}/${BNECK_OUTPUT}


run:
	docker run --name pydoop -p 8020:8020 -p 8042:8042 -p 8088:8088 -p 9000:9000 -p 10020:10020 -p 19888:19888 -p 50010:50010 -p 50020:50020 -p 50070:50070 -p 50075:50075 -p 50090:50090 -v ../pydeep:/mnt -d crs4/pydoop
	docker -it pydoop exec bash


clean:
	-echo "what?"

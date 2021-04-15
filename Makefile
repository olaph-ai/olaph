all: build generate push

build:
			docker build -q -t drozza/policy-generator:latest .

push:
			docker push -q drozza/policy-generator:latest

learn:
	    docker run -v $(shell pwd)/../tasks:/tasks -v $(shell pwd)/../models:/models \
								 -e TASKS_DIR=/tasks -e MODELS_DIR=/models \
	  								drozza/policy-generator:latest \
										FastLAS --d /tasks/synheart-controller-opa-istio.las > ../models/synheart-controller-opa-istio.lp

generate:
	    docker run -v $(shell pwd)/../tasks:/tasks -v $(shell pwd)/../models:/models \
								 -v $(shell pwd)/../policies:/policies -v $(shell pwd)/../data:/data \
								 -e DATA=synheart-controller-opa-istio.log -e DATA_DIR=/data \
								 -e TASKS_DIR=/tasks -e MODELS_DIR=/models \
	  								drozza/policy-generator:latest python3 /generator/main.py

bash:
			docker run -it drozza/policy-generator:latest bash

eval:
			opa eval -f pretty -i ../data/single/synheart-controller-opa-istio1.log.json -d ../policies/synheart-controller-opa-istio.rego "data.synheart_controller_opa_istio.allow"

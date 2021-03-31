all: build generate

build:
			docker build -q -t policy-generator .

generate:
	    docker run -v $(shell pwd)/../tasks:/tasks -v $(shell pwd)/../models:/models \
								 -v $(shell pwd)/../policies:/policies -v $(shell pwd)/../data:/data \
								 -e DATA=synheart-controller-opa-istio.log -e DATA_DIR=/data \
								 -e TASKS_DIR=/tasks -e MODELS_DIR=/models \
								 -e N_BODY_ATTRIBUTES=${N_BODY_ATTRIBUTES} \
	  								policy-generator python3 /generator/main.py

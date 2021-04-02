all: build generate

build:
			docker build -q -t policy-generator .

learn:
	    docker run -v $(shell pwd)/../tasks:/tasks -v $(shell pwd)/../models:/models \
								 -e TASKS_DIR=/tasks -e MODELS_DIR=/models \
	  								policy-generator \
										FastLAS --d /tasks/synheart-controller-opa-istio.las > ../models/synheart-controller-opa-istio.lp

generate:
	    docker run -v $(shell pwd)/../tasks:/tasks -v $(shell pwd)/../models:/models \
								 -v $(shell pwd)/../policies:/policies -v $(shell pwd)/../data:/data \
								 -e DATA=synheart-controller-opa-istio.log -e DATA_DIR=/data \
								 -e TASKS_DIR=/tasks -e MODELS_DIR=/models \
	  								policy-generator python3 /generator/main.py

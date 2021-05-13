all: build generate

build:
			docker build -q -t drozza/policy-generator:latest .

push:
			docker push -q drozza/policy-generator:latest

learn:
	    docker run -v $(shell pwd)/../tasks:/tasks -v $(shell pwd)/../models:/models \
								 -e TASKS_DIR=/tasks -e MODELS_DIR=/models \
	  								drozza/policy-generator:latest \
										FastLAS --d /tasks/synheart-controller-opa-istio1.las > ../models/synheart-controller-opa-istio1.lp

output:
	    docker run -v $(shell pwd)/../tasks:/tasks -v $(shell pwd)/../models:/models \
								 -e TASKS_DIR=/tasks -e MODELS_DIR=/models \
	  								drozza/policy-generator:latest \
										FastLAS --output-solve-program /tasks/synheart-controller-opa-istio.las > ../models/synheart-controller-opa-istio.lp

generate:
	    docker run -v $(shell pwd)/../tasks:/tasks -v $(shell pwd)/../models:/models \
								 -v $(shell pwd)/../policies:/policies -v $(shell pwd)/../data:/data \
								 -v $(shell pwd)/../diffs:/diffs -v $(shell pwd)/../plots:/plots \
							   -e CONFIG=/config/config.yaml \
	  								drozza/policy-generator:latest python3 /generator/main.py

distance:
			python3 $(shell pwd)/generator/distance.py

bash:
			docker run -v $(shell pwd)/../tasks:/tasks -v $(shell pwd)/../models:/models \
								 -v $(shell pwd)/../policies:/policies -v $(shell pwd)/../data:/data \
								 -v $(shell pwd)/../diffs:/diffs \
								 -it drozza/policy-generator:latest bash

eval:
			opa eval -f pretty -i ../data/single/synheart-controller-opa-istio1.log.json -d ../policies/synheart-controller-opa-istio0.rego "data.synheart_controller_opa_istio.allow"

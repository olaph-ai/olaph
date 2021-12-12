all: build generate

monitor: build mon

build:
			docker build -q -t drozza/olaph:latest .

push:
			docker push -q drozza/olaph:latest

learn:
	    docker run -v $(shell pwd)/../tasks:/tasks -v $(shell pwd)/../models:/models \
								 -e TASKS_DIR=/tasks -e MODELS_DIR=/models \
	  								drozza/olaph:latest \
										FastLAS --d /tasks/synth-heart9_4.las > ../models/synth-heart9_4.lp

output:
	    docker run -v $(shell pwd)/../tasks:/tasks -v $(shell pwd)/../models:/models \
								 -e TASKS_DIR=/tasks -e MODELS_DIR=/models \
	  								drozza/olaph:latest \
										FastLAS --output-solve-program /tasks/synheart-controller-opa-istio.las > ../models/synheart-controller-opa-istio.lp

generate:
	    docker run -v $(shell pwd)/../tasks:/tasks -v $(shell pwd)/../models:/models \
								 -v $(shell pwd)/../policies:/policies -v $(shell pwd)/../data:/data \
								 -v $(shell pwd)/../diffs:/diffs -v $(shell pwd)/../plots:/plots \
                 -v $(shell pwd)/config:/config -e CONFIG=/config/config.yaml \
	  								drozza/olaph:latest python3 /generator/main.py

distance:
			python3 $(shell pwd)/generator/distance.py

bash:
			docker run -v $(shell pwd)/../tasks:/tasks -v $(shell pwd)/../models:/models \
								 -v $(shell pwd)/../policies:/policies -v $(shell pwd)/../data:/data \
								 -v $(shell pwd)/../diffs:/diffs \
								 -it drozza/olaph:latest bash

# eval:
# 			opa eval -f pretty -i ../data/single/synheart-controller-opa-istio1.log.json -d ../policies/synheart-controller-opa-istio0.rego "data.synheart_controller_opa_istio.allow"
eval:
			opa eval -i ../data/single/synth-heart11-single.json -d ../policies/synth-heart11_1.rego "data.synth_heart11.allow"

mon:
	    docker run -it -v ${OLAPH_CONF}/tasks:/tasks -v ${OLAPH_CONF}/models:/models \
						         -v ${OLAPH_CONF}/policies:/policies \
							 -v ${OLAPH_CONF}/diffs:/diffs -v ${OLAPH_CONF}/plots:/plots \
                                                         -v $(shell pwd)/config:/config -e CONFIG=/config/config.yaml \
						         -v $${HOME}/.kube:/root/.kube \
	  							drozza/olaph:latest python3 /generator/active_monitoring.py

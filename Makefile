OLAPH_OUTDIR := $(shell pwd)

all: build gen

generate: build gen

monitor: build mon

build:
			docker build -t drozza/olaph:latest .

push:
			docker push drozza/olaph:latest

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

gen:
	    mkdir -p ${OLAPH_OUTDIR}/tasks ${OLAPH_OUTDIR}/models ${OLAPH_OUTDIR}/policies ${OLAPH_OUTDIR}/diffs ${OLAPH_OUTDIR}/plots && \
	    docker run -it -v ${OLAPH_OUTDIR}/tasks:/tasks \
	                   -v ${OLAPH_OUTDIR}/models:/models \
			   -v ${OLAPH_OUTDIR}/policies:/policies \
			   -v ${OLAPH_OUTDIR}/diffs:/diffs \
			   -v ${OLAPH_OUTDIR}/plots:/plots \
                           -v $(shell pwd)/config:/config -e CONFIG=/config/config_generate.yaml \
	  			drozza/olaph:latest python3 /generator/cli.py

distance:
			python3 $(shell pwd)/generator/distance.py

bash:
			docker run -v $(shell pwd)/../tasks:/tasks -v $(shell pwd)/../models:/models \
								 -v $(shell pwd)/../policies:/policies -v $(shell pwd)/../data:/data \
								 -v $(shell pwd)/../diffs:/diffs \
								 -it drozza/olaph:latest bash

eval:
			opa eval -i ../data/single/synth-heart11-single.json -d ../policies/synth-heart11_1.rego "data.synth_heart11.allow"

mon:
	    mkdir -p ${OLAPH_OUTDIR}/tasks ${OLAPH_OUTDIR}/models ${OLAPH_OUTDIR}/policies ${OLAPH_OUTDIR}/diffs ${OLAPH_OUTDIR}/plots && \
	    docker run -it -v ${OLAPH_OUTDIR}/tasks:/tasks \
	                   -v ${OLAPH_OUTDIR}/models:/models \
			   -v ${OLAPH_OUTDIR}/policies:/policies \
			   -v ${OLAPH_OUTDIR}/diffs:/diffs \
			   -v ${OLAPH_OUTDIR}/plots:/plots \
                           -v $(shell pwd)/config:/config -e CONFIG=/config/config_monitor.yaml \
			   -v $(HOME)/.kube:/root/.kube \
	  			drozza/olaph:latest python3 /generator/active_monitoring.py

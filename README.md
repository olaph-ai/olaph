# Olaph
Olaph learns enforceable policies of normal behaviour online from application logs. Walk-through demo: https://olaph-ai.github.io/.

Configuration:
* Environment variable `OLAPH_INDIR` - set to a directory that will contain Olaph's input data. Defaults to the current working directory.
* Environment variable `OLAPH_OUTDIR` - set to a directory that will contain Olaph's output. Defaults to the current working directory.
* If you are learning policies from custom data, set the `<data_path>` in `config/config.yaml` to the path to your data file in the `OLAPH_INDIR` directory. Defaults to `test-data.log` in `data/`. 
  ```
  paths:
    data: <data_path>
  ```

There are two modes for running Olaph. The [first mode](#learn-policies-from-a-log-file) is learning policies from a log file, where online learning is simulated by iterating through the file line-by-line. In the [second mode](#active-monitoring), Olaph is integrated into a Kubernetes cluster to learn policies for a live application.
## Learn policies from a log file
In this mode, Olaph can learn policies of normal behaviour from a file consisting of JSON logs on each line, like [test-data.log](https://github.com/olaph-ai/olaph/blob/main/data/test-data.log) in this repository.
### Prerequisites
Install Docker - [installation guide](https://docs.docker.com/get-docker/)
### Usage
Run Olaph on the provided dataset in this repository:
```
make generate
```
Watch the output of this command for prompts to relearn the policy:
```
Approve policy 2? y/n/relearn:
```
A policy difference should be created in the `diffs` folder in the current working directory, unless you have configured `OLAPH_OUTDIR` to point somewhere else.
Enter one of `y`, `n`, or change the parameters under `settings:` in `config/config_generate.yaml` and enter `relearn`. The policies themselves will be put in the `policies` directory. At the end of Olaph's execution, the policy confidence plot with the relearns and thresholds can be found in the `plots` directory.
## Active Monitoring
Olaph can be run as an active monitor, which learns online and enforces access policies for an application running on an Istio-enabled Kubernetes cluster.
### Prerequisites
Install Docker Desktop - [installation guide](https://docs.docker.com/get-docker/) and [enable the Kubernetes cluster](https://docs.docker.com/desktop/kubernetes/). Once the cluster is up, run the following commands.

To install Istio ([docs](https://istio.io/latest/docs/setup/getting-started/)):
```sh
curl -L https://istio.io/downloadIstio | ISTIO_VERSION=1.12.1 TARGET_ARCH=x86_64 sh -
cd istio-1.12.1
export PATH=$PWD/bin:$PATH
istioctl install --set profile=demo -y
```
To install the OPA Envoy Plugin ([docs](https://github.com/open-policy-agent/opa-envoy-plugin/tree/main/examples/istio#quick-start)):
```sh
kubectl apply -f https://raw.githubusercontent.com/olaph-ai/olaph/main/opa-istio.yaml
kubectl label namespace default opa-istio-injection="enabled"
kubectl label namespace default istio-injection="enabled"
```
Deploy the sample application:
```sh
kubectl apply -f https://raw.githubusercontent.com/istio/istio/master/samples/bookinfo/platform/kube/bookinfo.yaml
kubectl apply -f https://raw.githubusercontent.com/istio/istio/master/samples/bookinfo/networking/bookinfo-gateway.yaml
```
Deploy Olaph:
```sh
git clone https://github.com/olaph-ai/olaph.git
cd olaph
make monitor
```
The sample application should be accessible from a web browser using the url `http://localhost/productpage`.
### Usage
Interact with the sample application through the browser and keep an eye on Olaph's logs for policy relearns, which will output the relearned policy in the `policies` folder, along with the policy differences in `diffs`. A policy confidence graph will be outputted in the `plots` folder after suspending Olaph's execution with `CTRL+C`. A more detailed guide can be found at https://olaph-ai.github.io/.
## Uninstall
```sh
kubectl delete -f https://raw.githubusercontent.com/istio/istio/master/samples/bookinfo/platform/kube/bookinfo.yaml
kubectl delete -f https://raw.githubusercontent.com/istio/istio/master/samples/bookinfo/networking/bookinfo-gateway.yaml
kubectl label namespace default opa-istio-injection-
kubectl label namespace default istio-injection-
kubectl delete -f https://raw.githubusercontent.com/olaph-ai/olaph/main/opa-istio.yaml
```
[Uninstall Istio](https://istio.io/latest/docs/setup/getting-started/#uninstall)

[Uninstall Docker and Kubernetes](https://docs.docker.com/desktop/mac/install/#uninstall-docker-desktop)

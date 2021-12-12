# Olaph
Olaph learns enforceable policies of normal behaviour from application logs.
## Active Monitoring
Olaph can be run as an active monitor, which learns and enforces access policies for an application running on an Istio-enabled Kubernetes cluster.
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
In a new terminal window, deploy Olaph:
```sh
git clone https://github.com/olaph-ai/olaph.git
cd olaph
make monitor
```
In a new terminal window, expose the sample application:
```sh
minikube tunnel
```
Get the sample application url by:
```sh
export INGRESS_HOST=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
export INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].port}')
export GATEWAY_URL=$INGRESS_HOST:$INGRESS_PORT
echo "$GATEWAY_URL/productpage"    # Sample application url
```
Paste the output of the previous command into your browser, e.g. `127.0.0.1:80/productpage`
### Usage
Interact with the sample application through the browser and keep an eye on Olaph's logs for policy relearns, which will output the relearned policy in the `policies` folder, along with the policy differences in `diffs`. A policy confidence graph will be outputted in the `plots` folder after suspending Olaph's execution with `CTRL+C`.

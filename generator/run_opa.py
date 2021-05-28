from subprocess import run, CalledProcessError, DEVNULL
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s: %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def get_opa_denies(window, policy_path, package, restructure):
    rs = json.dumps(list(map(lambda r: r['input'], window)))
    denied_path = 'denied.rego'
    denied = f'''
package {package}
denied := [i | d := allow with input as input[i]; r := d.allowed; not r]
'''
    with open(denied_path, 'w') as f:
        f.write(denied)
    out = run(f'opa eval -f pretty -I -d {policy_path} -d {denied_path} "data.{package}.denied"',
              shell=True, check=True, text=True, capture_output=True, input=rs)
    denied_rs = [window[int(i)] for i in json.loads(out.stdout)]
    return len(denied_rs), denied_rs

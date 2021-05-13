from subprocess import run, CalledProcessError, DEVNULL
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s: %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def get_opa_denies(window, policy_path, package):
    rs = json.dumps(list(map(lambda r: r['input'], window)))
    allows = f'''
package {package}
allows := count([r | d := allow with input as input[_]; r := d.allowed; r])
'''
    with open('allows.rego', 'w') as f:
        f.write(allows)
    out = run(f'opa eval -f pretty -I -d {policy_path} -d allows.rego "data.{package}.allows"',
              shell=True, check=True, text=True, capture_output=True, input=rs)
    denies = len(window) - int(out.stdout)
    return denies

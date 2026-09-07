# Read IBM Quantum credentials from a local text file and save them, so the token never has to be typed
# on a command line. You create the file, you run this. Nothing here prints the token.
#
# 1. Make a file, for example ibm_creds.txt, with these lines (instance is optional):
#      token=YOUR_API_KEY
#      instance=YOUR_CRN
# 2. Run:
#      python setup_account.py ibm_creds.txt
# 3. Delete the file afterward. The account is now saved in your Qiskit config.
import sys


def parse_creds(path):
    # read key=value lines and return a dict. lines without = are ignored, # lines are comments.
    out = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip().lower()] = v.strip()
    return out


def main(argv):
    path = argv[0] if argv else "ibm_creds.txt"
    try:
        creds = parse_creds(path)
    except FileNotFoundError:
        print("file not found:", path)
        print("make a text file with a line token=YOUR_API_KEY, then run again.")
        return 1
    token = creds.get("token")
    instance = creds.get("instance") or None
    if not token:
        print("no token found in", path)
        print("the file needs a line like: token=YOUR_API_KEY")
        return 1

    from qiskit_ibm_runtime import QiskitRuntimeService
    kwargs = {"channel": "ibm_quantum_platform", "token": token, "overwrite": True}
    if instance:
        kwargs["instance"] = instance
    QiskitRuntimeService.save_account(**kwargs)
    print("account saved. you can delete", path, "now.")

    # confirm it loads, without printing anything secret.
    svc = QiskitRuntimeService()
    print("channel:", svc.channel)
    try:
        print("instances found:", len(svc.instances()))
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

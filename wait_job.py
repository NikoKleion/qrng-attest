# Wait for the submitted job to finish, checking every 30 seconds, then print the certificate.
# Light on the network: one status call per check. Gives up after one hour.
import sys
import time

from qiskit_ibm_runtime import QiskitRuntimeService

import poll_job


def main():
    name, jid = open("last_job.txt", encoding="utf-8").read().split()
    svc = QiskitRuntimeService()
    waited = 0
    while waited <= 3600:
        status = str(svc.job(jid).status())
        print("waited", waited, "s, status:", status, flush=True)
        if any(k in status.upper() for k in ("DONE", "ERROR", "CANCEL")):
            break
        time.sleep(30)
        waited += 30
    else:
        print("gave up after", waited, "s, job still not finished.")
        return 0
    print()
    return poll_job.main()


if __name__ == "__main__":
    sys.exit(main())

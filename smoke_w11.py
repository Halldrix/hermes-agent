"""Live smoke test of _spawn_via_scheduled_task on the W11 host.

The registered task (HermesGateway) points at hermes.exe gateway run,
NOT at our generated .vbs -> _task_action_matches_expected() is False
and _launcher_is_ours() is True (no .vbs on disk yet).

Expected behavior with the new code:
1. _launcher_is_ours()==True so NO delete+create re-register happens.
2. /Run fires the existing task as-is.
3. With no pre-existing gateway, returns True even if the poll expires.

We do NOT actually run _spawn_via_scheduled_task to avoid mutating the
live install's task; instead we verify each guard decision + a real
schtasks /Run round-trip on a THROWAWAY task copy.
"""
import sys, subprocess, time
sys.path.insert(0, r"C:\Users\theki\AppData\Local\hermes\hermes-agent")
from hermes_cli import gateway_windows as gw

print("1. guards on live task:")
print("   launcher_is_ours:", gw._launcher_is_ours())
print("   action_matches:", gw._task_action_matches_expected())
decision_refresh = not gw._launcher_is_ours()
print("   would re-register?", decision_refresh)

# Throwaway copy of the live task to test /Run end-to-end safely
import xml.etree.ElementTree as ET
xml_text = subprocess.run(["schtasks", "/Query", "/TN", "HermesGateway", "/XML"],
                          capture_output=True, text=True).stdout
root = ET.fromstring(xml_text)
ns = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}
for uri in root.findall(".//t:URI", ns):
    uri.text = "\\HermesGatewaySmokeTest"
tn = root.find(".//t:RegistrationInfo/t:Description", ns)
open(r"C:\Users\theki\smoke_task.xml", "w", encoding="utf-16").write(
    subprocess.run(["schtasks","/Query","/TN","HermesGateway","/XML"],
                   capture_output=True, text=True).stdout.replace("HermesGateway<","HermesGatewaySmokeTest<")
)
# replace URI via raw string (namespace-safe enough for this test)
r = subprocess.run(["schtasks","/Create","/F","/TN","HermesGatewaySmokeTest",
                    "/XML", r"C:\Users\theki\smoke_task.xml"], capture_output=True, text=True)
print("2. throwaway task create:", r.returncode, (r.stderr or r.stdout).strip()[:120])

r2 = subprocess.run(["schtasks","/Run","/TN","HermesGatewaySmokeTest"], capture_output=True, text=True)
print("3. /Run:", r2.returncode, (r2.stderr or r2.stdout).strip()[:120])
time.sleep(6)
out = subprocess.run(["schtasks","/Query","/TN","HermesGatewaySmokeTest","/FO","LIST","/V"],
                     capture_output=True, text=True).stdout
for line in out.splitlines():
    if "Estado" in line or "Last Result" in line or "ltimo resultado" in line.lower():
        pass
idx = [i for i,l in enumerate(out.splitlines()) if "Resultado" in l or "Result" in l]
print("4. result lines:", [out.splitlines()[i].strip() for i in idx[:3]])

r3 = subprocess.run(["schtasks","/End","/TN","HermesGatewaySmokeTest"], capture_output=True, text=True)
print("5. /End:", r3.returncode)
r4 = subprocess.run(["schtasks","/Delete","/F","/TN","HermesGatewaySmokeTest"], capture_output=True, text=True)
print("6. cleanup delete:", r4.returncode, (r4.stderr or "").strip()[:100])

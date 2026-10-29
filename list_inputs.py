import sys
import json
import sounddevice as sd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def get_input_devices():
    devices = sd.query_devices()
    hostapis = {i: api['name'] for i, api in enumerate(sd.query_hostapis())}
    input_devices = []
    seen = set()

    # Query default input device index
    default_dev_idx = None
    try:
        default_dev_idx = sd.default.device[0]
    except Exception:
        pass

    for idx, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            api_name = hostapis.get(d['hostapi'], '')
            name = d['name'].strip()

            # Skip raw WDM-KS driver paths and internal streaming artifacts
            if name.startswith("@") or "System32" in name or api_name == "Windows WDM-KS":
                continue
            if name in ("Primary Sound Capture Driver", "Microsoft Sound Mapper - Input"):
                continue

            # Unique identification key across APIs
            key = (name, api_name)
            if key not in seen:
                seen.add(key)
                is_default = (idx == default_dev_idx)
                label = f"{name} ({api_name})"
                if is_default:
                    label += " [Default]"
                input_devices.append({
                    "id": idx,
                    "name": name,
                    "api": api_name,
                    "channels": d['max_input_channels'],
                    "is_default": is_default,
                    "label": label
                })

    return input_devices


if __name__ == "__main__":
    devs = get_input_devices()
    print(json.dumps(devs, indent=2))


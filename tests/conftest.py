

import os as _os

# The harness runner waits for a quiet machine before each session; unit tests drive it
# against stubs and must not wait on whatever else the machine is doing.
_os.environ.setdefault("MCP_MIRROR_LOAD_GATE", "off")

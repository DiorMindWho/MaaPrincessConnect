import os
import sys

# utf-8 encoding for stdout
sys.stdout.reconfigure(encoding="utf-8")

current_file_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_file_path)
project_root_dir = os.path.dirname(current_script_dir)

if os.getcwd() != project_root_dir:
    os.chdir(project_root_dir)

if current_script_dir not in sys.path:
    sys.path.insert(0, current_script_dir)

from bootstrap import (
    check_and_install_dependencies,
    configure_initial_runtime_paths,
    ensure_venv_and_relaunch_if_needed,
)

configure_initial_runtime_paths(project_root_dir)
ensure_venv_and_relaunch_if_needed(current_file_path)
check_and_install_dependencies()

from maa.agent.agent_server import AgentServer
from maa.toolkit import Toolkit

import my_action
import my_reco
import check_equipment


def main():
    Toolkit.init_option("./")

    if len(sys.argv) < 2:
        print("Usage: python main.py <socket_id>")
        print("socket_id is provided by AgentIdentifier.")
        sys.exit(1)
        
    socket_id = sys.argv[-1]

    AgentServer.start_up(socket_id)
    AgentServer.join()
    AgentServer.shut_down()


if __name__ == "__main__":
    main()

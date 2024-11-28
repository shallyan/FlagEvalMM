import argparse
import subprocess
import requests
import time
import atexit
import json
import os.path as osp
from mmengine.config import Config
from flagevalmm.common.logger import get_logger
from flagevalmm.server.utils import get_random_port
from flagevalmm.registry import EVALUATORS, DATASETS
from flagevalmm.server.utils import maybe_register_class, merge_args
import os
import signal

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Infer a model")
    parser.add_argument("--tasks", nargs="+", required=True, help="tasks to run")
    parser.add_argument("--exec", type=str, help="model path in examples")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--try-run", action="store_true", help="try run mode")
    parser.add_argument("--output-dir", type=str, help="output dir")
    parser.add_argument("--data-root", type=str, help="data root")
    parser.add_argument("--model", type=str, help="model name or path")
    parser.add_argument("--cfg", type=str, help="config file")
    parser.add_argument("--num-workers", "--num-workers", type=int)
    parser.add_argument("--backend", type=str)
    parser.add_argument("--disable-evaluation-server", "-ds", action="store_true")
    parser.add_argument("--skip", action="store_true", help="skip finished tasks")
    parser.add_argument("--server-port", type=int, help="port of evaluation server")
    parser.add_argument(
        "--server-ip",
        type=str,
        default="http://localhost",
        help="ip of evaluation server",
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="quiet mode")
    parser.add_argument(
        "--without-infer", "-wi", action="store_true", help="without inference"
    )
    parser.add_argument("--url", type=str, help="url of api model")
    parser.add_argument("--api-key", type=str, help="api key of api model")
    parser.add_argument(
        "--use-cache", action="store_true", help="use cache of api model"
    )
    parser.add_argument(
        "--extra-args", type=str, help="extra args of local server model"
    )
    args = parser.parse_args()
    return args


def update_cfg_from_args(args):
    cfg = json.load(open(args.cfg)) if args.cfg else {}
    keys = ["num_workers", "backend", "url", "api_key", "use_cache", "extra_args"]
    for key in keys:
        if getattr(args, key):
            cfg[key] = getattr(args, key)
    if args.model:
        cfg["model_path"] = args.model
        cfg["model_name"] = args.model

    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    else:
        if args.cfg and "model_name" in cfg:
            model_name = cfg["model_name"].split("/")[-1]
            cfg["output_dir"] = cfg.get(
                "output_dir", f"{model_name}_{time.strftime('%Y%m%d_%H%M%S')}"
            )
        else:
            model_name = (
                args.model.split("/")[-1] if args.model else args.exec.split("/")[-1]
            )
            cfg["model_path"] = args.model
            cfg["output_dir"] = f"{model_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    return cfg


class ServerWrapper:
    def filter_finished_tasks(self, tasks, output_dir):
        finished_tasks = []
        for task_file in tasks:
            task_cfg = Config.fromfile(task_file)
            task_name = task_cfg.dataset.name
            if osp.exists(osp.join(output_dir, task_name, f"{task_name}.json")):
                logger.info(f"Task {task_name} already finished, skip")
                continue
            finished_tasks.append(task_file)
        return finished_tasks

    def __init__(self, args):
        self.args = args
        self.cfg = update_cfg_from_args(args)
        self.port = None
        self.evaluation_server_ip = args.server_ip
        self.evaluation_server = None
        self.evaluation_server_pid = None
        self.infer_process = None
        # Register cleanup at exit
        atexit.register(self.cleanup)

    def start(self):
        """Main method to start the server and run the model"""
        # Validate inputs
        assert self.args.exec is not None, "exec is required"

        # Handle finished tasks
        if self.args.skip:
            self.args.tasks = self.filter_finished_tasks(
                self.args.tasks, self.cfg["output_dir"]
            )
        if len(self.args.tasks) == 0:
            logger.info("No tasks to run, exit")
            return

        # Launch services
        self.maybe_launch_evaluation_server(self.args, self.cfg["output_dir"])
        self.run_model_adapter()

    def run_model_adapter(self):
        try:
            command = self._build_command()
            # Create a new process group
            self.infer_process = subprocess.Popen(
                command,
                preexec_fn=os.setsid if os.name != "nt" else None,
                creationflags=(
                    0 if os.name != "nt" else subprocess.CREATE_NEW_PROCESS_GROUP
                ),
            )
            logger.info(f"Started process with PID: {self.infer_process.pid}")
            self.infer_process.wait()
        except KeyboardInterrupt:
            logger.info("Received interrupt, cleaning up...")
            self.cleanup()
        finally:
            logger.info("Command execution finished.")

    def _build_command(self):
        """Private method to build the command for model execution"""
        command = []
        if self.args.exec.endswith("py"):
            assert osp.exists(self.args.exec), f"model path {self.args.exec} not found"
            command += [
                "python",
                self.args.exec,
                "--server_ip",
                self.evaluation_server_ip,
                "--server_port",
                str(self.port),
            ]
        else:
            assert osp.exists(
                f"{self.args.exec}/run.sh"
            ), f"run.sh not found in {self.args.exec}"
            command += [
                "bash",
                f"{self.args.exec}/run.sh",
                self.evaluation_server_ip,
                str(self.port),
            ]
        command.extend(["--cfg", f"{json.dumps(self.cfg)}"])
        return command

    def maybe_launch_evaluation_server(self, args, output_dir):
        if args.disable_evaluation_server:
            self.evaluation_server = None
            self.port = args.server_port if args.server_port else 5000
            return
        # choose a random port and make sure it's not occupied
        self.port = get_random_port()
        logger.info(f"Using port {self.port}")
        command = [
            "python",
            "flagevalmm/server/run_server.py",
            "--tasks",
            *args.tasks,
            "--output-dir",
            output_dir,
            "--port",
            str(self.port),
        ]

        if args.debug:
            command.append("--debug")
        if args.try_run:
            command.append("--try-run")
        if args.model:
            command.extend(["--checkpoint", args.model])
        if args.quiet:
            command.append("--quiet")

        self.evaluation_server = subprocess.Popen(command, start_new_session=True)
        # wait for server to start or timeout
        for _ in range(10):
            if self.is_server_running():
                break
            time.sleep(1)
        else:
            raise RuntimeError("Server failed to start")

        self.evaluation_server_pid = self.evaluation_server.pid

    def is_server_running(self):
        try:
            response = requests.get(
                f"{self.evaluation_server_ip}:{self.port}/task_info", timeout=10
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def cleanup(self):
        if self.infer_process:
            try:
                if os.name != "nt":  # Unix
                    os.killpg(os.getpgid(self.infer_process.pid), signal.SIGTERM)
                    # Add a wait time for the process to complete cleanup
                else:  # Windows
                    self.infer_process.terminate()

                try:
                    self.infer_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning("Process didn't terminate, forcing kill")
                    if os.name != "nt":
                        os.killpg(os.getpgid(self.infer_process.pid), signal.SIGKILL)
                    else:
                        self.infer_process.kill()
                    self.infer_process.wait()
            except Exception as e:
                logger.info(f"The process {self.infer_process.pid} is killed: {e}")
            finally:
                self.infer_process = None

        if not hasattr(self, "evaluation_server") or self.evaluation_server is None:
            return
        try:
            if self.is_server_running():
                # check if eval_finished
                while True:
                    response = requests.get(
                        f"{self.evaluation_server_ip}:{self.port}/eval_finished"
                    )
                    if response.json()["status"] == 1:
                        break
                    time.sleep(1)
                    logger.info("Waiting for eval to finish...")
                self.evaluation_server.terminate()
                self.evaluation_server.wait()
                logger.info(
                    f"Server with PID {self.evaluation_server_pid} has been terminated.\nPort {self.port} is released"
                )
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        logger.info("ServerWrapper cleanup completed")


def evaluate_only(args):
    cfg = update_cfg_from_args(args)

    for task_file in args.tasks:
        task_cfg = Config.fromfile(task_file)
        task_cfg = merge_args(task_cfg, task_file, args)
        maybe_register_class(task_cfg, task_file)
        if args.try_run:
            task_cfg.dataset.debug = True
        if "evaluator" in task_cfg:
            evaluator = EVALUATORS.build(task_cfg.evaluator)
            dataset = DATASETS.build(task_cfg.dataset)
            task_name = task_cfg.dataset.name
            if args.output_dir:
                output_dir = args.output_dir
            else:
                output_dir = osp.join(cfg["output_dir"], task_name)
            evaluator.process(dataset, output_dir, model_name=cfg.get("model_name", ""))
        else:
            logger.error(f"No evaluator found in config {task_file}")


def run():
    args = parse_args()
    if args.without_infer:
        evaluate_only(args)
    else:
        server = ServerWrapper(args)
        server.start()


if __name__ == "__main__":
    run()
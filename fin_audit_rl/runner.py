"""Запуск политики в отдельном процессе или контейнере с обменом через JSON.

Оценщик и AuditEnv находятся в основном процессе. Агент получает наблюдения
через stdin и возвращает действия через stdout. Для закрытой проверки
используется Docker, локальный режим предназначен для разработки.
"""
from __future__ import annotations

import json
import math
import os
import queue
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path


MAX_PROTOCOL_LINE = 256_000


class AgentProtocolError(RuntimeError):
    pass


def _docker_argv(command, image, solution_dir, container_name=None):
    if not image:
        raise ValueError("для запуска в Docker требуется указать образ")
    solution = str(Path(solution_dir).resolve())
    return [
        "docker", "run", "--rm", "-i",
        "--name", container_name or f"audit-agent-{uuid.uuid4().hex[:12]}",
        "--network", "none",
        "--pull=never",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--memory=4g",
        "--cpus=2",
        "--pids-limit=64",
        "--user=65534:65534",
        "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=64m",
        "--mount", f"type=bind,source={solution},target=/solution,readonly",
        "--workdir", "/solution",
        image, *command,
    ]


class _AgentProcess:
    def __init__(self, command, backend, image, solution_dir, cwd, env):
        if not command:
            raise ValueError("команда запуска агента должна быть непустой")
        if backend == "docker":
            self.container_name = f"audit-agent-{uuid.uuid4().hex[:12]}"
            argv = _docker_argv(command, image, solution_dir, self.container_name)
            cwd = None
            child_env = None
        elif backend == "local":
            argv = list(command)
            # Передаём дочернему процессу ограниченный набор переменных окружения.
            allowed = ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP")
            child_env = {k: os.environ[k] for k in allowed if k in os.environ}
            child_env.update({"PYTHONIOENCODING": "utf-8", "PYTHONNOUSERSITE": "1"})
            child_env.update(env or {})
        else:
            raise ValueError("backend должен принимать значение 'docker' или 'local'")
        if backend != "docker":
            self.container_name = None
        self.process = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=0, cwd=cwd, env=child_env,
        )
        self.lines = queue.Queue(maxsize=2)
        self.messages = queue.Queue(maxsize=1)
        self.stopped = threading.Event()
        for job in (self._read_stdout, self._write_stdin):
            threading.Thread(target=job, daemon=True).start()

    def _publish(self, item):
        while not self.stopped.is_set():
            try:
                self.lines.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def _read_stdout(self):
        try:
            buffer = b""
            while True:
                chunk = self.process.stdout.read(4096)
                if not chunk:
                    if buffer:
                        self._publish(AgentProtocolError("ответ агента должен завершаться переводом строки"))
                    else:
                        self._publish(None)
                    return
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if len(line) > MAX_PROTOCOL_LINE:
                        self._publish(AgentProtocolError("превышена допустимая длина строки протокола агента"))
                        return
                    self._publish(line.decode("utf-8"))
                if len(buffer) > MAX_PROTOCOL_LINE:
                    self._publish(AgentProtocolError("превышена допустимая длина строки протокола агента"))
                    return
        except (OSError, UnicodeError, ValueError) as exc:
            self._publish(exc)

    def _write_stdin(self):
        while not self.stopped.is_set():
            try:
                wire, sent = self.messages.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                while wire:
                    wire = wire[self.process.stdin.write(wire):]
            except (OSError, ValueError) as exc:
                sent.put(exc)
                return
            sent.put(None)

    def send(self, message, timeout):
        if self.process.poll() is not None:
            raise AgentProtocolError("процесс агента завершился")
        wire = json.dumps(message, allow_nan=False, separators=(",", ":")).encode("ascii") + b"\n"
        sent = queue.Queue(maxsize=1)
        self.messages.put_nowait((memoryview(wire), sent))
        try:
            failure = sent.get(timeout=max(0, timeout))
        except queue.Empty:
            raise TimeoutError("истекло время передачи данных агенту") from None
        if failure:
            raise AgentProtocolError("агент закрыл входной поток протокола") from failure

    def receive(self, timeout):
        try:
            item = self.lines.get(timeout=max(0.001, timeout))
        except queue.Empty:
            raise TimeoutError("истекло время ожидания действия агента") from None
        if isinstance(item, BaseException):
            raise AgentProtocolError(f"ошибка чтения стандартного вывода агента: {type(item).__name__}: {item}") from item
        if item is None:
            raise AgentProtocolError("агент закрыл выходной поток протокола")
        try:
            message = json.loads(item, parse_constant=_invalid_json_constant, parse_float=_finite_float)
        except (ValueError, RecursionError):
            raise AgentProtocolError("стандартный вывод агента содержит некорректный JSON") from None
        if not isinstance(message, dict) or not isinstance(message.get("action"), dict):
            raise AgentProtocolError("ожидается JSON-ответ {'action': {...}}")
        pending = [(message["action"], 0)]
        while pending:
            value, depth = pending.pop()
            if depth > 32:
                raise AgentProtocolError("превышена допустимая глубина вложенности действия агента")
            if isinstance(value, dict):
                pending.extend((child, depth + 1) for child in value.values())
            elif isinstance(value, list):
                pending.extend((child, depth + 1) for child in value)
        return message["action"]

    def stop(self):
        self.stopped.set()
        if self.process.poll() is None:
            if self.container_name:
                try:
                    subprocess.run(["docker", "kill", self.container_name],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5, check=False)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        for stream in (self.process.stdin, self.process.stdout):
            if stream and not stream.closed:
                stream.close()


def _invalid_json_constant(_value):
    raise ValueError("JSON содержит бесконечное число или NaN")


def _finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        _invalid_json_constant(value)
    return result


def run_isolated_episode(
    env,
    command,
    *,
    seed=0,
    config=None,
    backend="docker",
    image=None,
    solution_dir=None,
    cwd=None,
    child_env=None,
    allow_insecure_local=False,
):
    """Запустить эпизод агента с передачей наблюдений через JSON-протокол.

    Сначала основной процесс отправляет {"event":"reset","observation":...},
    затем чередует ответы step_result с действиями агента {"action": {...}}.
    Метрики возвращаются вызывающей стороне оценщика. Награда передаётся
    агенту при env.mode == "train".
    """
    if env.mode == "eval" and backend != "docker" and not allow_insecure_local:
        raise ValueError("закрытая оценка требует backend='docker', локальный режим предназначен для разработки")
    observation, _ = env.reset(seed, config)
    agent = _AgentProcess(command, backend, image, solution_dir, cwd, child_env)
    status = "completed"

    def remaining():
        return env.max_seconds - (time.monotonic() - env.started)

    try:
        agent.send({"event": "reset", "observation": observation}, remaining())
        while not env.done:
            if remaining() <= 0:
                raise TimeoutError("истекло время ожидания действия агента")
            action = agent.receive(remaining())
            observation, reward, terminated, truncated, _ = env.step(action)
            result = {"event": "step_result", "observation": observation,
                      "terminated": terminated, "truncated": truncated}
            if env.mode == "train":
                result["reward"] = reward
            try:
                agent.send(result, remaining())
            except (TimeoutError, AgentProtocolError):
                if not (terminated or truncated):
                    raise
            if truncated and remaining() <= 0:
                status = "agent_timeout"
    except TimeoutError:
        status = "agent_timeout"
        env.abort()
    except AgentProtocolError:
        status = "agent_protocol_error"
        env.abort(error="ошибка протокола агента")
    finally:
        agent.stop()
    return {
        "status": status,
        "episode_id": env.episode["manifest"]["episode_id"],
        "evaluator_metrics": env.evaluator_result(),
        "reward": env.evaluator_reward,
        "trajectory": env.trajectory,
    }


def docker_available():
    """Проверить наличие команды Docker в PATH."""
    return shutil.which("docker") is not None

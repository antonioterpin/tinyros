# 🤖 TinyROS - A minimal operating systems for Robots

[![GitHub stars](https://img.shields.io/github/stars/antonioterpin/tinyros?style=social)](https://github.com/antonioterpin/tinyros/stargazers)
[![Code Style](https://github.com/antonioterpin/tinyros/actions/workflows/code-style.yaml/badge.svg)](https://github.com/antonioterpin/tinyros/actions/workflows/code-style.yaml)
[![PyPI version](https://img.shields.io/pypi/v/tinyros.svg)](https://pypi.org/project/tinyros)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

A minimal implementation of an operating system for robots, to ease the integration of sensors, actuators, and heavy compute directly in the physical world.

## 🏛️ Design Philosophy

🪶 **Single-file solution**: `TinyROS` ships its own minimal transport — an RPC-style pub/sub wire on top of `AF_UNIX` (TCP loopback on Windows) with a `multiprocessing.shared_memory` side-channel for large ndarray payloads. It works under the assumption that in most robotic systems, communication is primarily peer-to-peer or involves only a few subscribers per publisher (in [ROS](https://docs.ros.org/) terminology). This targeted approach lets us strip complexity down significantly: we deliberately avoid the entire ROS ecosystem baggage while providing the familiar publisher-subscriber pattern for the 90% of use cases that don't need the full complexity of ROS2.

✨ **Cross-platform and easy to install**: `TinyROS` comes without installation headaches and is extremely lean while being cross-platform. You can develop on macOS, Windows, Linux, etc. It maintains the same (or better) efficiency as ROS2 implementations while being completely written in Python. We increase flexibility, ease of use, clarity, and reduce package size without compromising performance.

🎯 **Static configuration over dynamic discovery**: Unlike traditional ROS systems that rely on dynamic node discovery and runtime topic resolution, `TinyROS` deliberately enforces a static network configuration defined upfront. This design choice **is a feature, not a bug**. By requiring explicit declaration of all nodes, topics, and connections in a YAML configuration file, we achieve:

- *Clarity*: The entire system topology is visible at a glance
- *Predictability*: No surprises from nodes appearing or disappearing at runtime
- *Debugging*: Easy to trace data flow and identify connection issues
- *Documentation*: The network config serves as living documentation of your system
- *Reliability*: Eliminates race conditions and discovery-related failures

We believe that for most robotics applications, the network topology is known at design time and changes infrequently. Embracing this reality leads to simpler, more robust systems.

## 🏗️ Projects Built with TinyROS

`TinyROS` has been used in the following robotics projects:
[![FluidsControl](https://img.shields.io/badge/GitHub-antonioterpin%2Ffluidscontrol-2ea44f?logo=github)](https://github.com/antonioterpin/fluidscontrol)

If you use `TinyROS` in your project, please open a PR to add it here 🤗.

## 🚀 Quick Start
For once, this is going to be painless 🤗.

### Installation

With `uv`:
```bash
uv add tinyros
```
With `pip`:
```bash
pip install tinyros
```

For the installation from source or for development, please see our [Contributing Guide](./CONTRIBUTING.md).

### Supported Platforms 💻

| Linux | macOS | Windows|
|---|---|---|
|✅|✅|n/a (likely)|

## 🔥 Examples
A full example is available in `main.py`.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for detailed information on:
- Development workflow and branch management
- Code style requirements and automated checks
- Testing standards and coverage expectations
- PR preparation and commit message conventions

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

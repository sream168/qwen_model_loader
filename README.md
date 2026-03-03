# Qwen Offline Model Loader

本地离线 GGUF 模型推理服务，支持 Qwen 3.5 等最新模型架构，提供 OpenAI / Ollama 兼容 API。

## 特性

- **双引擎架构**：默认使用 `llama-server` 子进程引擎（支持最新 GGUF 架构），可选 `llama-cpp-python` 绑定引擎
- **OpenAI 兼容**：`/v1/chat/completions`（支持流式 SSE）
- **Ollama 兼容**：`/api/chat`、`/api/generate`（支持流式 NDJSON）
- **通用接口**：`/health`、`/models`、`/v1/models`、`/api/tags`
- **配置管理**：`/config` 获取和动态更新模型映射及推理参数
- **LRU 模型缓存**：自动管理内存，淘汰时优雅停止引擎子进程
- **生命周期管理**：应用退出时自动清理所有子进程资源

## 项目结构

```
qwen_model_loader/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI 应用工厂 + lifespan 管理
│   ├── config_store.py          # 线程安全的配置存储（原子性写入）
│   ├── model_manager.py         # 模型 LRU 缓存 + 引擎工厂
│   ├── service.py               # 业务逻辑层
│   ├── schemas.py               # Pydantic 数据模型
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py            # 所有 API 端点
│   └── engine/
│       ├── base.py              # 引擎抽象基类
│       ├── llama_server_engine.py   # llama-server 子进程引擎（默认）
│       └── llama_engine.py      # llama-cpp-python 绑定引擎（可选）
├── config/
│   └── config.json              # 默认配置文件
├── tests/
│   └── test_core.py             # 单元测试
└── pyproject.toml               # 项目配置与依赖
```

---

## 环境搭建（Linux / Ubuntu）

### 1. 系统依赖

```bash
sudo apt-get update
sudo apt-get install -y cmake build-essential git python3 python3-venv
```

### 2. 编译 llama.cpp（获取 llama-server）

`llama-cpp-python` 最新版（0.3.16）不支持 Qwen 3.5 的 `qwen35` 架构。llama.cpp 本身从 b8149 版本开始已支持，因此通过编译 `llama-server` 二进制来解决。

```bash
# 克隆 llama.cpp 最新源码
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /home/admin/llama.cpp

# cmake 配置（Release 模式）
cmake -S /home/admin/llama.cpp -B /home/admin/llama.cpp/build -DCMAKE_BUILD_TYPE=Release

# 编译 llama-server（使用全部 CPU 核心并行编译）
cmake --build /home/admin/llama.cpp/build --target llama-server -j$(nproc)

# 验证编译结果
/home/admin/llama.cpp/build/bin/llama-server --version
# 预期输出: version: 1 (ecd99d6)
```

编译完成后，`llama-server` 位于 `/home/admin/llama.cpp/build/bin/llama-server`。

> **GPU 加速（可选）**：如需 CUDA 支持，在 cmake 配置时添加 `-DGGML_CUDA=ON`：
> ```bash
> cmake -S /home/admin/llama.cpp -B /home/admin/llama.cpp/build \
>   -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
> ```

### 3. 下载模型文件

将 GGUF 格式的模型文件放到模型目录（默认 `/home/admin/Downloads`）：

```bash
# 示例：从 Hugging Face 下载 Qwen3.5-0.8B 量化版
# huggingface-cli download Qwen/Qwen3.5-0.8B-GGUF Qwen3.5-0.8B-Q4_K_M.gguf --local-dir /home/admin/Downloads
ls -lh /home/admin/Downloads/Qwen3.5-0.8B-Q4_K_M.gguf
```

### 4. 安装项目

```bash
cd /home/admin/qwen_model_loader

# 创建并激活虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装项目及测试依赖
pip install -e ".[test]"
```

### 5. 启动服务

```bash
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 30007
```

### 6. 验证

```bash
# 健康检查
curl http://localhost:30007/health

# 非流式推理
curl -X POST http://localhost:30007/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"你好"}]}'

# 流式推理
curl -N -X POST http://localhost:30007/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"你好"}],"stream":true}'
```

---

## 环境搭建（Windows）

### 1. 安装前置工具

- **Git**：https://git-scm.com/download/win（安装时勾选"Add to PATH"）
- **Python 3.10+**：https://www.python.org/downloads/（安装时勾选"Add to PATH"）
- **CMake**：https://cmake.org/download/（安装时选择"Add CMake to the system PATH"）
- **Visual Studio Build Tools**：https://visualstudio.microsoft.com/visual-cpp-build-tools/
  - 安装时选择 **"使用 C++ 的桌面开发"** 工作负载

### 2. 编译 llama.cpp（获取 llama-server.exe）

打开 **"x64 Native Tools Command Prompt for VS"**（从开始菜单搜索）：

```cmd
:: 克隆源码
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git C:\llama.cpp

:: cmake 配置
cmake -S C:\llama.cpp -B C:\llama.cpp\build -DCMAKE_BUILD_TYPE=Release

:: 编译 llama-server
cmake --build C:\llama.cpp\build --target llama-server --config Release

:: 验证
C:\llama.cpp\build\bin\Release\llama-server.exe --version
```

> **GPU 加速（可选）**：安装 [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 后，cmake 配置时添加 `-DGGML_CUDA=ON`。

### 3. 下载模型文件

将 GGUF 模型文件放到指定目录，例如 `C:\Models\`：

```
C:\Models\Qwen3.5-0.8B-Q4_K_M.gguf
```

### 4. 安装项目

打开 PowerShell 或 CMD：

```powershell
cd C:\qwen_model_loader

# 创建虚拟环境
python -m venv .venv
.\.venv\Scripts\activate

# 安装项目
pip install -e ".[test]"
```

### 5. 修改配置

编辑 `config/config.json`，调整为 Windows 路径：

```json
{
  "model_dir": "C:\\Models",
  "model_mapping": {
    "Qwen3.5-0.8B": "Qwen3.5-0.8B-Q4_K_M.gguf"
  },
  "default_model": "Qwen3.5-0.8B",
  "generation_defaults": {
    "temperature": 0.7,
    "max_tokens": 512,
    "top_p": 0.9
  },
  "max_loaded_models": 1,
  "engine_type": "llama_server",
  "llama_server_path": "C:\\llama.cpp\\build\\bin\\Release\\llama-server.exe",
  "n_ctx": 4096,
  "n_threads": 8,
  "startup_timeout": 60
}
```

### 6. 启动服务

```powershell
.\.venv\Scripts\activate
uvicorn app.main:app --host 0.0.0.0 --port 30007
```

### 7. 验证

```powershell
# PowerShell
Invoke-RestMethod -Uri http://localhost:30007/health

# 推理测试
$body = '{"messages":[{"role":"user","content":"你好"}]}'
Invoke-RestMethod -Uri http://localhost:30007/v1/chat/completions `
  -Method Post -ContentType "application/json" -Body $body
```

---

## 配置说明

默认配置文件 `config/config.json`，首次启动自动生成。可通过 API 动态修改。

```json
{
  "model_dir": "/home/admin/Downloads",
  "model_mapping": {
    "Qwen3.5-0.8B": "Qwen3.5-0.8B-Q4_K_M.gguf"
  },
  "default_model": "Qwen3.5-0.8B",
  "generation_defaults": {
    "temperature": 0.7,
    "max_tokens": 512,
    "top_p": 0.9
  },
  "max_loaded_models": 1,
  "engine_type": "llama_server",
  "llama_server_path": "/home/admin/llama.cpp/build/bin/llama-server",
  "n_ctx": 4096,
  "n_threads": 8,
  "startup_timeout": 60
}
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_dir` | string | `/home/admin/Downloads` | 模型文件存放目录 |
| `model_mapping` | object | - | 模型别名 → GGUF 文件名映射 |
| `default_model` | string | `Qwen3.5-0.8B` | 请求未指定模型时使用的默认模型 |
| `generation_defaults` | object | - | 推理参数默认值 |
| `max_loaded_models` | int | `1` | 最多同时缓存的模型数量（1-4） |
| `engine_type` | string | `llama_server` | 引擎类型：`llama_server` 或 `llama_cpp` |
| `llama_server_path` | string | - | llama-server 二进制文件路径 |
| `n_ctx` | int | `4096` | 上下文窗口大小（512-131072） |
| `n_threads` | int | `8` | 推理线程数（1-128） |
| `startup_timeout` | int | `60` | llama-server 启动超时秒数（5-300） |

### 引擎类型选择

| 引擎 | 优势 | 限制 |
|------|------|------|
| `llama_server`（默认） | 支持最新模型架构，独立于 Python 绑定版本 | 需要编译 llama.cpp |
| `llama_cpp` | 无需额外编译，纯 Python 安装 | 受限于 llama-cpp-python 发布节奏 |

---

## API 参考

### 通用接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/models` | 模型列表（OpenAI 格式） |
| GET | `/v1/models` | 模型列表（OpenAI 格式，别名） |
| GET | `/api/tags` | 模型列表（Ollama 格式） |
| GET | `/config` | 获取当前配置 |
| POST | `/config` | 更新配置（部分更新） |

### 推理接口

| 方法 | 路径 | 格式 | 流式 |
|------|------|------|------|
| POST | `/v1/chat/completions` | OpenAI | SSE (`text/event-stream`) |
| POST | `/api/chat` | Ollama | NDJSON (`application/x-ndjson`) |
| POST | `/api/generate` | Ollama | NDJSON (`application/x-ndjson`) |

### 请求示例

#### OpenAI Chat Completions

```bash
# 非流式
curl http://localhost:30007/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.5-0.8B",
    "messages": [{"role":"user","content":"你好"}],
    "stream": false
  }'

# 流式
curl -N http://localhost:30007/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.5-0.8B",
    "messages": [{"role":"user","content":"你好"}],
    "stream": true
  }'
```

#### Ollama Chat

```bash
curl http://localhost:30007/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.5-0.8B",
    "messages": [{"role":"user","content":"你好"}],
    "stream": false
  }'
```

#### Ollama Generate

```bash
curl http://localhost:30007/api/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.5-0.8B",
    "prompt": "什么是量子计算？",
    "stream": false
  }'
```

#### 更新配置

```bash
curl -X POST http://localhost:30007/config \
  -H 'Content-Type: application/json' \
  -d '{
    "generation_defaults": {"temperature": 0.5, "max_tokens": 1024},
    "engine_type": "llama_server",
    "n_ctx": 8192
  }'
```

---

## 测试

```bash
source .venv/bin/activate
pip install -e ".[test]"
pytest -q
```

---

## 常见问题

### Q: 启动时报 "llama-server 二进制文件未找到"

确认 `config/config.json` 中 `llama_server_path` 指向正确的 `llama-server` 路径，并确认已完成编译。

### Q: 模型加载失败，报 "Model load failed"

- 使用 `llama_server` 引擎可解决大部分架构不支持的问题
- 确认 GGUF 文件完整且未损坏
- 确认 llama.cpp 版本支持该模型架构

### Q: 如何切换回 llama-cpp-python 引擎？

```bash
# 安装 llama-cpp-python
pip install llama-cpp-python

# 修改配置
curl -X POST http://localhost:30007/config \
  -H 'Content-Type: application/json' \
  -d '{"engine_type": "llama_cpp"}'
```

### Q: Windows 上编译 llama.cpp 报错

确保使用 **"x64 Native Tools Command Prompt for VS"** 而非普通 CMD/PowerShell。该命令行提供了正确的 MSVC 编译器环境变量。

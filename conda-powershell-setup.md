# Conda PowerShell Setup for RyzenAI-SW

This guide shows how to use conda with PowerShell for the RyzenAI-SW project without hooking system PowerShell.

## Simply use the full path

```powershell
& "C:\path\to\miniforge3\Scripts\conda.exe" activate ryzen-ai-1.6.1
```

environment activated:
(ryzen-ai-1.6.1) PS C:\path\to\MyRyzenAI\RyzenAI-SW>

## Managing Other Environments

```powershell
# List all environments
& "C:\path\to\miniforge3\Scripts\conda.exe" env list

# Activate different environment
& "C:\path\to\miniforge3\Scripts\conda.exe" activate hello_world_env

# Deactivate current environment
& "C:\path\to\miniforge3\Scripts\conda.exe" deactivate

# Create new environment
& "C:\path\to\miniforge3\Scripts\conda.exe" create -n myenv python=3.11

# Install packages
& "C:\path\to\miniforge3\Scripts\conda.exe" install numpy
```

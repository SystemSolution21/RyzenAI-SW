# Conda PowerShell Setup for RyzenAI-SW

This guide shows how to set up conda to work properly with PowerShell for the RyzenAI-SW project without hijacking your system PowerShell.

## Problem

- Manual `conda activate` doesn't work without proper initialization
- `conda init pwsh` hijacks PowerShell by auto-activating the base environment
- Need conda functionality only when working on RyzenAI-SW project

## Solution

### Step 1: Initialize Conda for PowerShell

```powershell
conda init powershell
```

**What this does:** Sets up the necessary conda functions (`activate`, `deactivate`) in PowerShell profile.

### Step 2: Disable Auto-Activation

```powershell
conda config --set auto_activate_base false
```

**What this does:** Prevents conda from automatically activating the `(base)` environment when starting PowerShell.

### Step 3: Restart Terminal

Close and reopen your terminal (or restart VS Code) for changes to take effect.

### Step 4: Manual Activation (When Needed)

```powershell
conda activate ryzen-ai-1.6.1
```

**Result:** You should see the environment name in your prompt:

```powershell
(ryzen-ai-1.6.1) PS path\to\RyzenAI-SW\tutorial\getting_started_resnet\int8>
```

## Benefits

- ✅ PowerShell works normally (no hijacking)
- ✅ Conda activation works properly when needed
- ✅ No auto-activation of base environment
- ✅ Clean system PowerShell experience
- ✅ Full conda functionality available on demand

## Usage for RyzenAI-SW

1. Open VS Code terminal in your RyzenAI-SW project
2. Run: `conda activate ryzen-ai-1.6.1`
3. Work with your project using the conda environment
4. Use `conda deactivate` when done (optional)

## Verification

After setup, you should be able to:

- Use PowerShell normally without conda interference
- Manually activate conda environments with proper prompt display
- See Visual Studio 2022 Developer Command Prompt initialization when activating ryzen-ai-1.6.1

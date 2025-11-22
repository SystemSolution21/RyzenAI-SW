# Comprehensive Explanation of `hello_world.py`

This script demonstrates the **complete workflow** for running a PyTorch model on AMD Ryzen AI NPU. It's a tutorial that shows the entire pipeline from model creation to NPU acceleration.

---

## 📋 **Overview**

The script performs these main steps:

1. **Create** a PyTorch CNN model
2. **Export** to ONNX format
3. **Quantize** the model for NPU compatibility
4. **Run** on CPU (baseline)
5. **Run** on NPU (accelerated)
6. **Compare** performance

---

## 🔍 **Detailed Breakdown**

### **Part 1: Model Definition (Lines 19-44)**

```python
class SmallModel(nn.Module):
    def __init__(self):
        super(SmallModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = torch.add(x, 1)
        return x
```

**Purpose**: Creates a simple CNN with 4 convolutional layers

- **Input**: 3-channel image (RGB), 224×224 pixels
- **Architecture**:
  - Conv1: 3→32 channels
  - Conv2: 32→64 channels
  - Conv3: 64→128 channels
  - Conv4: 128→256 channels
- **Activation**: ReLU after each conv layer
- **Final operation**: Adds 1 to all values

---

### **Part 2: ONNX Export (Lines 54-74)**

```python
dummy_input = torch.rand(batch_size, input_channels, input_size, input_size)

torch.onnx.export(
    pytorch_model,
    (dummy_input,),
    tmp_model_path,
    export_params=True,
    opset_version=17,  # Recommended opset
    input_names=["input"],
    output_names=["output"],
    dynamic_axes=dynamic_axes,
)
```

**Purpose**: Converts PyTorch model to ONNX format

- **Why ONNX?** NPU requires ONNX format (not PyTorch)
- **Dummy input**: Used to trace the model's computation graph
- **Dynamic axes**: Allows variable batch size (though NPU uses batch=1)
- **Opset 17**: ONNX operator set version (recommended for Ryzen AI)

**Key Parameters**:

- `export_params=True`: Include trained weights in the ONNX file
- `input_names/output_names`: Names for model inputs/outputs
- `(dummy_input,)`: Tuple format required by torch.onnx.export

---

### **Part 3: Quantization (Lines 77-100)**

```python
quant_config = get_default_config("XINT8")
quant_config.extra_options["UseRandomData"] = True
config = Config(global_quant_config=quant_config)

quantizer = ModelQuantizer(config)
quant_model = quantizer.quantize_model(
    model_input=input_model_path,
    model_output=output_model_path,
    calibration_data_path=None,
)
```

**Purpose**: Converts FP32 model to INT8 for NPU efficiency

**Key Concepts**:

- **XINT8**: Mixed precision quantization
  - Weights: INT8 (8-bit signed integers)
  - Activations: UINT8 (8-bit unsigned integers)
- **Why quantize?**
  - NPU is optimized for INT8 operations
  - Reduces model size by ~4x
  - Increases inference speed
  - Minimal accuracy loss
- **UseRandomData**: Uses random calibration data (for demo purposes)
- **Cross-Layer Equalization (CLE)**: Optimizes quantization across layers

**Quantization Process**:

1. Analyzes activation ranges using calibration data
2. Determines optimal scaling factors
3. Converts FP32 weights/activations to INT8
4. Inserts QuantizeLinear/DequantizeLinear nodes in ONNX graph

---

### **Part 4: CPU Inference (Lines 102-125)**

```python
cpu_session = onnxruntime.InferenceSession(
    model.SerializeToString(),
    providers=["CPUExecutionProvider"],
    sess_options=cpu_options,
)

start = timer()
cpu_results = cpu_session.run(None, {"input": input_data})
cpu_total = timer() - start
```

**Purpose**: Runs the quantized model on CPU as a baseline

- **ONNX Runtime**: Microsoft's inference engine
- **CPUExecutionProvider**: Runs on CPU cores
- **Timing**: Measures execution time for comparison
- **Input data**: Random numpy array matching model input shape

---

### **Part 5: NPU Detection (Lines 127-168)**

```python
command = r"pnputil /enum-devices /bus PCI /deviceids "
process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()

# Check for supported Hardware IDs
if "PCI\\VEN_1022&DEV_1502&REV_00" in stdout.decode(encoding="utf-8", errors="ignore"):
    npu_type = "PHX/HPT"
if "PCI\\VEN_1022&DEV_17F0&REV_00" in stdout.decode(encoding="utf-8", errors="ignore"):
    npu_type = "STX"
```

**Purpose**: Detects which AMD NPU is installed

**Supported NPUs**:

| NPU Type | Device ID | Processor Series | xclbin File |
|----------|-----------|------------------|-------------|
| **PHX/HPT** (Phoenix/Hawk Point) | `1502` | Ryzen 7040/8040 series | `phoenix/4x4.xclbin` |
| **STX** (Strix Point) | `17F0` | Ryzen AI 300 series | `strix/AMD_AIE2P_4x4_Overlay.xclbin` |

**xclbin file**: Binary configuration file that programs the NPU hardware architecture

**Configuration Selection**:

```python
match npu_type:
    case "PHX/HPT":
        xclbin_file = os.path.join(install_dir, "voe-4.0-win_amd64", "xclbins", "phoenix", "4x4.xclbin")
    case "STX":
        xclbin_file = os.path.join(install_dir, "voe-4.0-win_amd64", "xclbins", "strix", "AMD_AIE2P_4x4_Overlay.xclbin")
```

---

### **Part 6: Cache Management (Lines 170-181)**

```python
directory_path = os.path.join(current_directory, r"cache\hello_cache")

if os.path.exists(directory_path):
    shutil.rmtree(directory_path)
    print(f"Directory '{directory_path}' deleted successfully.")
```

**Purpose**: Clears compilation cache to force fresh compilation

- **Why?** Ensures you're testing the latest model version
- **Cache location**: `cache/hello_cache/`
- **What's cached?** Compiled NPU binaries from previous runs
- **When to clear**: When model changes or debugging compilation issues

---

### **Part 7: NPU Inference (Lines 184-208)**

```python
aie_options = onnxruntime.SessionOptions()
provider_options = [{}]

if npu_type == "PHX/HPT":
    # For PHX/HPT devices, xclbin is required
    provider_options = [{"target": "X1", "xclbin": xclbin_file}]

aie_session = onnxruntime.InferenceSession(
    model.SerializeToString(),
    providers=["VitisAIExecutionProvider"],
    sess_options=aie_options,
    provider_options=provider_options,
)

start = timer()
npu_results = aie_session.run(None, {"input": input_data})
npu_total = timer() - start
```

**Purpose**: Runs the model on NPU

**Key Components**:

- **VitisAIExecutionProvider**: ONNX Runtime plugin for AMD NPU
- **Provider options**:
  - PHX/HPT needs `xclbin` file path and `target: "X1"`
  - STX uses default configuration (empty dict)
- **Compilation**: First run compiles model for NPU (takes longer)
- **Subsequent runs**: Use cached compilation (much faster)

**What Happens During First Run**:

1. Model graph is analyzed
2. Operations are partitioned (NPU vs CPU)
3. NPU subgraphs are compiled to NPU binary
4. Compiled binary is cached for future use

---

### **Part 8: Performance Comparison (Lines 210-212)**

```python
print(f"CPU Execution Time: {cpu_total}")
print(f"NPU Execution Time: {npu_total}")
```

**Purpose**: Shows speedup from NPU acceleration

**Example Results**:

- CPU: 0.149 seconds
- NPU: 0.023 seconds
- **Speedup: ~6.6x faster on NPU!** 🚀

---

## 🎯 **Key Takeaways**

### **The Complete Pipeline**

```
PyTorch Model → ONNX Export → INT8 Quantization → NPU Compilation → Inference
```

### **Why Each Step Matters**

1. **ONNX Export**: NPU doesn't understand PyTorch directly
2. **Quantization**: NPU is optimized for INT8, not FP32
3. **NPU Detection**: Different NPUs need different configurations
4. **VitisAI EP**: Bridges ONNX Runtime to AMD NPU hardware

### **Performance Benefits**

- ✅ **6-7x faster** inference on NPU vs CPU
- ✅ **Lower power consumption** (NPU is more efficient)
- ✅ **Frees up CPU/GPU** for other tasks
- ✅ **Enables real-time AI** on laptops

---

## 🔧 **Important Configuration Details**

### **Model Requirements for NPU**

- ✅ Batch size = 1 (NPU doesn't support batching)
- ✅ Fixed input shapes (no dynamic dimensions)
- ✅ INT8 quantization
- ✅ ONNX opset 17 recommended
- ✅ Supported operations (Conv2D, MatMul, etc.)

### **Environment Variables**

- `RYZEN_AI_INSTALLATION_PATH`: Points to Ryzen AI SDK installation
  - Example: `C:\Ryzen-AI\ryzen-ai-sw-1.3`

### **Files Generated**

| File | Description | Size |
|------|-------------|------|
| `models/helloworld.onnx` | Original FP32 ONNX model | ~10 MB |
| `models/helloworld_quantized.onnx` | INT8 quantized model | ~3 MB |
| `cache/hello_cache/` | Compiled NPU binaries | Auto-generated |

---

## 💡 **What This Demonstrates**

This is a **minimal working example** showing:

- How to prepare models for AMD Ryzen AI NPU
- The complete toolchain: PyTorch → ONNX → Quantization → NPU
- Performance benefits of NPU acceleration
- Hardware detection and configuration
- Proper use of ONNX Runtime execution providers

It's the foundation for running more complex models (ResNet, BERT, LLMs, etc.) on Ryzen AI NPU!

---

## 🐛 **Common Issues and Solutions**

### **Issue 1: Type Error on torch.onnx.export**

```
Argument of type "dict[str, Tensor]" cannot be assigned to parameter "args"
```

**Solution**: Pass inputs as a tuple, not a dictionary:

```python
# ❌ Wrong
torch.onnx.export(model, {"x": dummy_input}, ...)

# ✅ Correct
torch.onnx.export(model, (dummy_input,), ...)
```

### **Issue 2: Custom Ops Library Warning**

```
[QUARK-WARNING]: The custom ops library does NOT exist.
```

**Solution**: This is non-critical. Install Visual Studio C++ Build Tools if you want to compile custom ops, but quantization works without it.

### **Issue 3: NPU Not Detected**

```
Unrecognized APU type. Exiting.
```

**Solution**:

- Verify you have a Ryzen AI processor (7040/8040/AI 300 series)
- Update chipset drivers
- Check Device Manager for NPU device

### **Issue 4: Slow First Run**

**Solution**: This is normal! First run compiles the model for NPU. Subsequent runs use cached compilation and are much faster.

---

## 📚 **Further Reading**

- [AMD Ryzen AI Documentation](https://ryzenai.docs.amd.com/)
- [ONNX Runtime VitisAI Execution Provider](https://onnxruntime.ai/docs/execution-providers/Vitis-AI-ExecutionProvider.html)
- [Quark Quantization Library](https://github.com/amd/quark)
- [PyTorch ONNX Export Guide](https://pytorch.org/docs/stable/onnx.html)

---

## 🎓 **Next Steps**

After understanding this hello world example, you can:

1. **Try more complex models**: ResNet, MobileNet, BERT
2. **Use real calibration data**: Replace `UseRandomData` with actual dataset
3. **Optimize for accuracy**: Fine-tune quantization parameters
4. **Deploy in applications**: Integrate NPU inference into your apps
5. **Benchmark different models**: Compare NPU vs CPU/GPU performance

---

**Created**: 2025-11-22
**Ryzen AI SDK Version**: 1.3 (voe-4.0)
**ONNX Runtime**: 1.23.2
**Quark Version**: 0.10

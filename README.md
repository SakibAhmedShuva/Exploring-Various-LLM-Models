# Exploring Various LLM Models

A comprehensive exploration of open-source Large Language Models available on Hugging Face Hub. This repository demonstrates how to work with different LLM architectures, implement them locally, and compare their performance across various tasks.

## 🎯 Overview

This repository provides hands-on experience with various open-source LLMs from Hugging Face, including model loading, inference, fine-tuning, and optimization techniques. Perfect for researchers, developers, and AI enthusiasts who want to explore the capabilities of different open-source language models.

## 📚 Table of Contents

- [Models Explored](#models-explored)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Notebook Examples](#notebook-examples)
- [Model Formats](#model-formats)
- [Performance Comparisons](#performance-comparisons)
- [Usage Examples](#usage-examples)
- [Contributing](#contributing)
- [License](#license)

## 🤖 Models Explored

### Large Language Models
- **LLaMA 2** (7B, 13B, 70B) - Meta's open-source foundation models
- **Mistral 7B** - Efficient high-performance model
- **Zephyr 7B** - Fine-tuned Mistral for chat applications
- **Code Llama** (7B, 13B, 34B) - Specialized for code generation
- **Vicuna** - Fine-tuned LLaMA for conversations
- **WizardLM** - Instruction-following model variants
- **OpenHermes** - Enhanced conversational capabilities
- **Nous Hermes** - Improved reasoning and chat performance

### Quantized Models (GGUF Format)
- **MiMo-7B-RL-GGUF** - Reinforcement Learning enhanced model
- Various quantization levels (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0)
- CPU-optimized inference with llama.cpp compatibility

### Specialized Models
- **CodeT5+** - Code understanding and generation
- **StarCoder** - Multi-language code generation
- **WizardCoder** - Enhanced coding capabilities
- **Phind CodeLlama** - Search-augmented coding assistant

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (8GB+ VRAM recommended)
- At least 16GB system RAM
- Git LFS for large model files

### Installation

```bash
# Clone the repository
git clone https://github.com/SakibAhmedShuva/Exploring-Various-LLM-Models.git
cd Exploring-Various-LLM-Models

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate bitsandbytes
pip install datasets evaluate rouge-score
pip install jupyter notebook ipywidgets

# For GGUF model support
pip install llama-cpp-python
pip install ctransformers[cuda]  # For GPU acceleration
```

## 📓 Notebook Examples

### Current Notebooks

- **MiMo-7B-RL-GGUF.ipynb** - Exploring MiMo 7B model with reinforcement learning enhancements in GGUF format

### Planned Notebooks

- **model_comparison.ipynb** - Side-by-side comparison of different LLMs
- **quantization_analysis.ipynb** - Performance impact of different quantization methods
- **fine_tuning_guide.ipynb** - Fine-tuning smaller models on custom datasets
- **inference_optimization.ipynb** - Speed and memory optimization techniques

## 🔧 Model Formats

### Standard PyTorch Models
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
```

### GGUF Models (CPU Optimized)
```python
from llama_cpp import Llama

# Load GGUF model for CPU inference
llm = Llama(
    model_path="./models/mimo-7b-rl.Q4_0.gguf",
    n_ctx=4096,
    n_threads=8
)

# Generate text
output = llm("Explain quantum computing:", max_tokens=200)
print(output['choices'][0]['text'])
```

### 4-bit Quantized Models
```python
from transformers import BitsAndBytesConfig

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
```

## 📊 Performance Comparisons

### Model Specifications

| Model | Parameters | Quantization | VRAM Usage | Inference Speed | Use Case |
|-------|------------|-------------|------------|----------------|----------|
| Mistral 7B | 7.3B | FP16 | ~14GB | Fast | General purpose |
| Mistral 7B | 7.3B | 4-bit | ~4GB | Medium | Resource-limited |
| LLaMA 2 7B | 6.7B | FP16 | ~13GB | Medium | Conversation |
| Code Llama 7B | 6.7B | FP16 | ~13GB | Medium | Code generation |
| MiMo 7B (GGUF) | 7B | Q4_0 | ~4GB | Fast (CPU) | CPU inference |

### Benchmark Tasks

- **Text Generation Quality** - Creative writing, storytelling
- **Code Generation** - Python, JavaScript, SQL queries  
- **Question Answering** - Factual and reasoning questions
- **Instruction Following** - Complex multi-step tasks
- **Conversation** - Multi-turn dialogue quality

## 💡 Usage Examples

### Basic Text Generation

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_text(model_name, prompt, max_length=200):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
prompt = "Explain machine learning in simple terms:"
response = generate_text("mistralai/Mistral-7B-v0.1", prompt)
print(response)
```

### Model Comparison Script

```python
def compare_models(prompt, models_list):
    results = {}
    
    for model_name in models_list:
        print(f"Testing {model_name}...")
        try:
            response = generate_text(model_name, prompt)
            results[model_name] = response
        except Exception as e:
            results[model_name] = f"Error: {str(e)}"
    
    return results

# Compare multiple models
models_to_test = [
    "mistralai/Mistral-7B-v0.1",
    "meta-llama/Llama-2-7b-hf",
    "HuggingFaceH4/zephyr-7b-beta"
]

comparison = compare_models("Write a short story about AI:", models_to_test)
```

## 🛠️ Repository Structure

```
Exploring-Various-LLM-Models/
├── notebooks/
│   ├── MiMo-7B-RL-GGUF.ipynb
│   ├── model_comparison.ipynb (planned)
│   └── quantization_analysis.ipynb (planned)
├── models/
│   └── downloaded_models/  # Local model storage
├── scripts/
│   ├── download_models.py
│   ├── benchmark_models.py
│   └── convert_to_gguf.py
├── utils/
│   ├── model_loader.py
│   ├── evaluation_metrics.py
│   └── optimization.py
├── requirements.txt
├── LICENSE
└── README.md
```

## 🔍 Key Features

- **Model Exploration**: Hands-on experience with various open-source LLMs
- **Format Comparison**: PyTorch, GGUF, and quantized model formats
- **Performance Analysis**: Memory usage, speed, and quality comparisons
- **Optimization Techniques**: Quantization, CPU inference, GPU optimization
- **Practical Examples**: Real-world use cases and implementation patterns

## 🤝 Contributing

Contributions are welcome! Areas where you can help:

- Adding new model explorations
- Creating comparison benchmarks
- Optimizing inference code
- Adding new notebook examples
- Improving documentation

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-model`)
3. Add your notebook or code
4. Commit changes (`git commit -m 'Add exploration of NewModel-7B'`)
5. Push to branch (`git push origin feature/new-model`)
6. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Useful Resources

- [Hugging Face Model Hub](https://huggingface.co/models)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [llama.cpp Repository](https://github.com/ggerganov/llama.cpp)
- [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

## 📞 Contact

- **Author**: Sakib Ahmed Shuva  
- **GitHub**: [@SakibAhmedShuva](https://github.com/SakibAhmedShuva)

---

**⭐ Star this repository if you find it helpful for your LLM exploration journey!**

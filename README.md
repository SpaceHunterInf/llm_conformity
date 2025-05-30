# Conformity in Large Language Models

![Main Figure](img/main.png)

This repository contains the code and experiments for our paper "Conformity in Large Language Models". We investigate how Large Language Models (LLMs) exhibit conformity behavior when presented with incorrect majority opinions in multi-party dialogue scenarios.

## 🔍 Overview

This research examines conformity effects in Large Language Models by simulating multi-party dialogues where participants provide responses to various questions. We test whether LLMs conform to incorrect majority answers across different domains including:

- **Multiple Choice QA**: MMLU, CommonsenseQA
- **Fact Checking**: Symmetric dataset with contradictory evidence
- **Subjective QA**: Political opinion questions (Politiscale)
- **Numerical Reasoning**: BBH Object Counting
- **Open-ended QA**: PopQA

## 🛠 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/llm_conformity.git
cd llm_conformity

# Install dependencies
pip install -r requirements.txt
```

## 📁 Repository Structure

```
llm_conformity/
├── data/                          # Dataset files
│   ├── MMLU_Test/                # MMLU test data
│   ├── PopQA/                    # PopQA dataset
│   ├── Politiscale/              # Political opinion questions
│   └── BIG-Bench-Hard/           # BBH Object Counting
├── utils/                         # Utility functions
│   ├── prompt_utils.py           # Prompt generation
│   ├── dataset_utils.py          # Dataset loading
│   ├── eval_utils.py             # Evaluation metrics
│   └── aux_utils.py              # Auxiliary functions
├── evaluate/                      # Custom evaluation library
├── img/                          # Figures and visualizations
├── vllm_mpd.py                   # Main inference script
├── vllm_settings_eval.py         # Experiment runner
└── README.md
```

## 🚀 Quick Start

### Basic Usage

Run conformity experiments on CommonsenseQA:

```bash
python vllm_settings_eval.py \
    --dataset_name CommonsenseQA \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --max_participants 10 \
    --conform_mode all \
    --participant_mode plain
```

### Advanced Configuration

```bash
python vllm_settings_eval.py \
    --dataset_name MMLU_Test \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --max_participants 10 \
    --conform_mode all \
    --participant_mode plain \
    --save_mcqa_first_prob \
    --log_probs 5
```

## 📊 Datasets

### Supported Datasets

1. **MMLU** - Multiple choice questions across academic subjects
2. **CommonsenseQA** - Commonsense reasoning questions
3. **PopQA** - Popular factual questions with distractors
4. **Politiscale** - Political opinion questions (subjective)
5. **BBH Object Counting** - Numerical reasoning tasks
6. **Symmetric Dataset** - Fact-checking with contradictory evidence

### Dataset Loading

```python
from utils.dataset_utils import *

# Load MMLU data
data = get_MMLU_data('data/MMLU_Test/astronomy_test.csv')

# Load CommonsenseQA
data = get_CommensenseQA_data(split='validation')

# Load PopQA
data = get_popqa_data('data/PopQA/popqa_with_distractors.jsonl')
```

## 🧪 Evaluation

### Conformity Metrics

The evaluation framework measures:
- **Conformity Rate**: Percentage of times the model changes to the majority (incorrect) answer
- **Accuracy Retention**: How often the model maintains correct answers
- **Response Distribution**: Analysis of "other" and "no answer" responses

### Running Evaluations

```python
from utils.eval_utils import *

# Evaluate MCQA conformity
columns, conformed, correct, others, no_answer, total = mcqa_eval(data, max_participants=10)

# Plot results
plot_conformity(participants, correct, conformed, others, no_answer, 'output.jpg', 'Experiment Title')

# Analyze token probabilities
first_probs, conform_probs, correct_probs = mcqa_logits_eval(data, max_participants=10)
plot_logits_trend(first_probs, conform_probs, correct_probs, 'trend.jpg', 'Probability Trends')
```

## 🔬 Experiments

### Experiment Modes

1. **Conform Mode**:
   - `all`: All previous participants give the **same** incorrect answers 
   - `fixed_length_all`: Question Distillation setup
   - `one_other_plus`: Devil's Advocate setup
   Detailed setups can be found in `utils/prompt_utils.py`
2. **Participant Mode**:
   - `plain`: Simple participant representation
   - `confident`: Participants express their answers in confident tones.
   - `uncertain`: Participants express their answers in uncertain tones.
   Detailed prompts can be found in `utils/*_.templates.txt`

### Example Experiment

```python
from vllm_mpd import *

# Initialize model
llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct", dtype='float32')
sampling_params = SamplingParams(temperature=0, top_p=1.0, max_tokens=50)

# Run base cycle (individual responses)
data = mcqa_base_cycle(llm, sampling_params, data)

# Run conformity cycle (multi-party dialogue)
data = mcqa_conform_cycle(llm, sampling_params, data, 
                         max_participants=10, 
                         conform_mode='all', 
                         participant_mode='plain')
```

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Citation

If you use this code or find our work helpful, please cite:

```bibtex
@misc{zhu2025conformitylargelanguagemodels,
      title={Conformity in Large Language Models}, 
      author={Xiaochen Zhu and Caiqi Zhang and Tom Stafford and Nigel Collier and Andreas Vlachos},
      year={2025},
      eprint={2410.12428},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.12428}, 
}
```

---

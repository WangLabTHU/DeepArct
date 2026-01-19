# 🧬🧱 DeepArct: Autonomous Synthesis of Regulatory Sequence Models


**DeepArct** is a hierarchical, knowledge-grounded multi-agent system designed to automate the end-to-end synthesis of deep learning pipelines for regulatory genomics. It addresses the "human-centric bottleneck" in computational biology by transforming days of manual literature review and iterative architecture design into a **5-to-10 minute automated workflow**.


---

## 🌟 Why DeepArct?

* **Data-Adaptive:** Directly maps your specific dataset characteristics (e.g., sequence length, noise profile, data modality) to architectural requirements.
* **Knowledge-Grounded:** Powered by a dual-layer knowledge base—integrating the full texts of **28 foundational papers** with thousands of scientific abstracts via RAG.
* **Transparent & Interpretable:** Shifting from "black-box" optimization to **constraint-driven synthesis**. Every model choice is backed by a traceable biological rationale (e.g., Reverse-Complement symmetry, receptive field calibration).
* **Zero Manual Tuning:** Delivers a complete, executable pipeline including data preprocessing, optimized model architecture, and rigorous training/evaluation protocols.

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and set up the environment:

```
git clone [https://github.com/your-username/DeepArct.git](https://github.com/your-username/DeepArct.git)
cd DeepArct
conda create -n DeepArct python=3.11
conda activate DeepArct
pip install -r requirements.txt
```

### 2. Configuration
Configure your LLM API credentials in the environment file:
```
Bash

# Edit /Agents/.env
OPENAI_API_KEY=your_api_key_here
# Supporting OpenAI, Claude, or compatible API providers
```

### 3. Define Your Task
Fill in your modeling intent in /task/task_description.json. Using our Standardized Prompt Template:
```
JSON

{
    "task_name": "",
    "task_goal": "",
    "task_requirements": "",
    "task_dataset": {
        "file_path": "",
        "description": "",
        "columns": { },
        "input_features": "",
        "target_variable": "",
        "key_constraints": "",
    }
}

```

## 🛠 Workflow
DeepArct operates in two distinct phases:

### Phase 1: Collaborative Design
Run the multi-agent deliberation process. The Data, Method, Model, and Result Expert agents will collaborate to synthesize a comprehensive design strategy.
```
Bash

python run.py
Output: Check /outputs/ for the Designing Report and raw Agent Conversation Logs. You can audit the rationales or intervene to steer the design process.
```

### Phase 2: Code Synthesis
Once the design is finalized, generate the complete, executable Python modeling pipeline:
```
Bash

python code_multi.py
```
Output: Fully functional .py files including data loaders, model classes (e.g., hybrid CNN-Transformer backbones), and training loops tailored to your specific hardware and data regime.

## 📂 Project Structure
/Agents/: Core logic and prompt engineering for the specialized expert agents.

/Knowledge_Base/: Curated foundational papers and RAG-retrieval mechanisms.

/task/: User input templates and task specifications.

/outputs/: Generated design rationales, conversation logs, and diagnostic reports.

/generated_code/: The final, synthesized deep learning pipelines ready for training.

## 📦 Dataset and Trained Models

The datasets and trained models are archived on Zenodo at: https://zenodo.org/records/18270213.

Usage Instructions: To facilitate model execution, please place the downloaded files in ./data  into the ./task/ directory within the DeepArct repository. This configuration ensures that the automated workflow correctly identifies the biological sequences and corresponding regulatory labels during the synthesis process.



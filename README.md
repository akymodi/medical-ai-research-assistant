# 🏥 Medical AI Research Assistant

A fine-tuned GPT-Neo 2.7B model trained on PubMed and ClinicalTrials.gov data for medical research assistance.

## 🎯 Overview

This project fine-tunes a large language model (GPT-Neo 2.7B) on medical research data to create an intelligent assistant capable of:
- Answering medical research questions
- Analyzing clinical trial data
- Providing evidence-based medical insights
- Searching and referencing medical literature

## 🚀 Model Information

- **Base Model**: [EleutherAI GPT-Neo 2.7B](https://huggingface.co/EleutherAI/gpt-neo-2.7B)
- **Fine-tuned Model**: [abmodi/pubmed-2.5B](https://huggingface.co/abmodi/pubmed-2.5B)
- **Parameters**: 2.7 billion
- **Training Data**: 702 medical research articles and clinical trials
- **Training Device**: Apple M3 Ultra with MPS acceleration

## 📊 Dataset

The model was trained on a comprehensive medical dataset including:
- **PubMed Articles**: Research papers from various medical journals
- **Clinical Trials**: Data from ClinicalTrials.gov
- **Categories**: Multiple medical specialties and research areas
- **Time Range**: 2024 data with historical context

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch
- Transformers library
- SQLite3

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/medical-ai-research-assistant.git
cd medical-ai-research-assistant

# Install dependencies
pip install -r requirements.txt

# Download the fine-tuned model (optional - will download automatically)
# The model is available on Hugging Face: abmodi/pubmed-2.5B
```

## 🎮 Usage

### 1. Interactive Medical Assistant
```bash
python3 medical_ai_assistant_gpt_neo.py
```

Start an interactive chat session where you can ask medical questions and get AI-powered responses.

### 2. Model Testing
```bash
python3 test_gpt_neo_model.py
```

Run comprehensive tests to evaluate model performance on various medical questions.

### 3. Database Management
```bash
python3 database.py
```

Manage your medical research database, add new articles, and search existing data.

## 📁 Project Structure

```
medical-ai-research-assistant/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── database.py                         # Database management
├── pubmed_api.py                       # PubMed API integration
├── clinicaltrials_api.py               # ClinicalTrials.gov API
├── fine_tune_gpt_neo_2_7b_simple.py   # Model fine-tuning script
├── medical_ai_assistant_gpt_neo.py     # Interactive assistant
├── test_gpt_neo_model.py              # Model testing suite
├── pubmed_data.db                      # SQLite database
├── gpt-neo-2.7b-medical-finetuned/    # Fine-tuned model files
└── docs/                              # Documentation
    ├── GPT_NEO_2_7B_SUCCESS_REPORT.md
    └── LARGE_MODELS_SUMMARY.md
```

## 🧪 Model Performance

The fine-tuned model demonstrates:
- **Training Loss**: Converged from 2.16 to 1.45
- **Generation Speed**: ~30 seconds per response on M3 Ultra
- **Medical Accuracy**: High relevance for medical terminology
- **Context Understanding**: Good comprehension of medical research context

## 🔧 Fine-tuning Process

The model was fine-tuned using:
- **Framework**: Hugging Face Transformers
- **Training Strategy**: Causal language modeling
- **Optimization**: AdamW optimizer with cosine learning rate
- **Hardware**: Apple M3 Ultra with MPS acceleration
- **Memory Management**: Gradient checkpointing and accumulation

## 📚 API Integration

### PubMed API
- Searches medical literature from PubMed
- Retrieves abstracts, authors, and metadata
- Supports date range and keyword filtering

### ClinicalTrials.gov API
- Accesses clinical trial data
- Retrieves trial phases, status, and outcomes
- Supports condition and intervention filtering

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Areas for Contribution
- Additional medical data sources
- Model performance improvements
- User interface enhancements
- Documentation improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [EleutherAI](https://www.eleuther.ai/) for the GPT-Neo model
- [Hugging Face](https://huggingface.co/) for the Transformers library
- [PubMed](https://pubmed.ncbi.nlm.nih.gov/) for medical literature data
- [ClinicalTrials.gov](https://clinicaltrials.gov/) for clinical trial data

## 📞 Contact

- **GitHub**: [@yourusername](https://github.com/yourusername)
- **Hugging Face**: [@abmodi](https://huggingface.co/abmodi)
- **Email**: your.email@example.com

## 🏆 Achievements

✅ Successfully fine-tuned 2.7B parameter model on medical data  
✅ Trained on Apple Silicon with MPS acceleration  
✅ Published model on Hugging Face Hub  
✅ Created comprehensive medical AI system  
✅ Built interactive assistant and testing tools  

## 🔮 Future Roadmap

- [ ] Web interface for the medical assistant
- [ ] API endpoints for model integration
- [ ] Support for additional medical data sources
- [ ] Model quantization for mobile deployment
- [ ] Multi-language support for medical terminology
- [ ] Integration with electronic health records

---

**⭐ If you find this project helpful, please give it a star!**

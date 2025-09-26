# <div align="center">TrustJudge: Inconsistencies of LLM-as-a-Judge and How to Alleviate Them</div>

<div align="center">
<a href="https://arxiv.org/abs/2509.21117" target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv></a>
<a href="https://github.com/TrustJudge/TrustJudge/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green"></a>
<a><img alt="Static Badge" src="https://img.shields.io/badge/made_with-Python-blue"></a>
</div>

For the LLM-as-a-judge evaluation setting, this library systematically addresses two long-standing consistency issues—Score–Comparison inconsistency (lower-rated responses winning in pairwise comparisons) and Pairwise Transitivity inconsistency (e.g., A>B>C yet C>A). It implements TrustJudge, a probabilistic evaluation framework that: (1) uses distribution-sensitive scoring to convert discrete rating probabilities into a continuous expectation, preserving information entropy for finer scores; and (2) applies likelihood-aware aggregation to resolve transitivity conflicts via bidirectional preference probabilities or perplexity.

**If you have any question, feel free to contact YunzeSong77@gmail.com and yidongwang37@gmail.com**

## Install environment

Clone the Repository and Install the Packages:

```bash
git clone https://github.com/TrustJudge/TrustJudge
cd TrustJudge
pip install -r requirements.txt
```

## Usage

### 1. Data Demo

Here we provide a demo of human-annotated data:
[`data/answers/filtered_selected_answers.jsonl`](./data/answers/filtered_selected_answers.jsonl)
[`data/answers/filtered_selected_answers_with_category.jsonl`](./data/answers/filtered_selected_answers_with_category.jsonl)

**We will upload full data for reproduction soon.**

---

### 2. Pipeline Demo

Here we provide the script to run the full end-to-end pipeline (generate single-score → generate pairwise comparison → calculate inconsistency metrics).

```sh
bash scripts/demo.sh
```

For step-by-step commands and intermediate data, see [pipeline details](./detail.md).


## Citation

If you find this repository useful, please cite our work.

```
@misc{wang2025trustjudge,
      title={TrustJudge: Inconsistencies of LLM-as-a-Judge and How to Alleviate Them}, 
      author={Wang, Yidong and Song, Yunze and Zhu, Tingyuan and Zhang, Xuanwang and Yu, Zhuohao and Chen, Hao and Song, Chiyu and Wang, Qiufeng and Wang, Cunxiang and Wu, Zhen and Dai, Xinyu and Zhang, Yue and Ye, Wei and Zhang, Shikun},
      year={2025},
      eprint={2509.21117},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2509.21117}, 
}
```

## License

TrustJudge is licensed under the [MIT License](./LICENSE).

# 🧠 智能文本分类系统 (BERT + 随机森林)

本项目是一个功能完备的中文文本分类系统，结合了深度学习 **BERT** 模型与传统机器学习 **随机森林** 模型。系统提供美观的 **Streamlit** 前端交互界面和实时的 **Flask** 后端预测 API。

---

## ✨ 核心特性

- **双引擎支持**: 默认使用 BERT 深度学习模型（F1-score: **96.37%**），支持扩展随机森林模型。
- **现代化 UI**: 基于 Streamlit 构建，采用 **Glassmorphism (玻璃拟态)** 设计风格，支持深色模式、动态卡片和响应式布局。
- **实时预测**: 后端基于 Flask 毫秒级响应，支持海量文本并发预测。
- **历史追踪**: 自动保存预测历史，支持示例快速输入。
- **完整类别**: 支持 10 大新闻类别分类（科技、教育、游戏、时尚、财经、家居、娱乐、体育、房产、时政）。

---

## 📂 项目结构

```text
03-bert/
├── bert-base-chinese/      # 预训练 BERT 模型权重
├── save_models/            # 训练好的模型权重 (.pt 文件)
├── scripts/                # 辅助/演示脚本 (单机预测、量化等)
├── a1_dataloader_utils.py  # 数据加载、预处理及标签映射 (LABEL_MAP)
├── a2_bert_classifer_model.py # BERT 模型结构定义
├── a3_train.py             # 模型训练脚本 (1 Epoch 即可达 96%+ 准确率)
├── api_flask_server.py     # 后端 Flask API 服务入口
├── app_streamlit2.py       # 前端 Streamlit 交互平台
├── bert_predict_func.py    # 核心预测解码算法
├── config.py               # 项目全局参数配置
├── model2dev_utils.py      # 模型评估与验证工具
└── requirements.txt        # 项目依赖清单
```

---

## 🛠️ 环境准备

建议使用 **Conda** 创建虚拟环境：

```powershell
# 创建环境
conda create -n tmf python=3.10
conda activate tmf

# 安装依赖
pip install -r requirements.txt
```

---

## 🚀 快速启动

启动系统需要两个独立终端窗口：

### 1. 启动后端 API
```powershell
conda activate tmf
python api_flask_server.py
```
*服务启动后将监听 `http://localhost:8010`*

### 2. 启动交互 UI
```powershell
conda activate tmf
streamlit run app_streamlit2.py --server.port 8888
```
*访问 **http://localhost:8888** 即可开始使用。*

---

## 📊 模型表现

| 指标 | BERT 模型结果 |
| :--- | :--- |
| **准确率 (Accuracy)** | 96.37% |
| **F1-Score** | 96.37% |
| **平均预测耗时** | ~50ms |

---

## 📝 开发者指南

### 如何修改类别顺序
如果需要调整分类类别（如增加新类别）：
1. 修改 `01-data/class.csv`。
2. 同步更新 `a1_dataloader_utils.py` 中的 `LABEL_MAP` 的索引。
3. 运行 `python a3_train.py` 重新训练以使模型索引生效。

---

## 📜 许可证

本代码仅供科研与论文参考使用。使用 BERT 模型请遵循相关开源协议。

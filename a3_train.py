import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
from tqdm import tqdm
from model2dev_utils import model2dev
from config import Config
import warnings
# 导入数据处理工具类
from a1_dataloader_utils import build_dataloader
# 导入bert模型
from a2_bert_classifer_model import BertClassifier

warnings.filterwarnings("ignore")

# 加载配置对象，包含模型参数、路径等
conf = Config()


def model2train():
    """
    训练 BERT 分类模型并在验证集上评估，保存最佳模型。
    参数：无显式参数，所有配置通过全局 conf 对象获取。
    返回：无返回值，训练过程中保存最佳模型到指定路径。
    """
    # 准备数据
    train_dataloader, dev_dataloader, test_dataloader = build_dataloader(
    train_path=conf.train_datapath,  # 假设 Config 中定义了训练集路径
    dev_path=conf.dev_datapath,      # 验证集路径（原参数）
    test_path=conf.test_datapath     # 假设 Config 中定义了测试集路径
)
    # 准备模型
    # 初始化bert分类模型
    model = BertClassifier()
    # 将模型移动到指定的设备
    model.to(conf.device)

    # 准备损失函数
    loss_fn = nn.CrossEntropyLoss()

    # 准备优化器
    optimizer = AdamW(model.parameters(), lr=conf.learning_rate)

    # 开始训练模型
    # 初始化F1分数，用于保存最好的模型
    best_f1 = 0.0
    # 外层循环遍历每个训练轮次
    #  （每次需要设置训练模式，累计损失，预存训练集测试和真实标签）
    for epoch in range(conf.num_epochs):
        # 设置模型为训练模式
        model.train()
        # 初始化累计损失，初始化训练集预测和真实标签
        total_loss = 0.0
        train_preds, train_labels = [], []
        # 内层循环遍历训练DataLoader每个批次
        for i, batch in enumerate(tqdm(train_dataloader, desc="训练集训练中...")):
            # 提取批次数据并移动到设备
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(conf.device)
            attention_mask = attention_mask.to(conf.device)
            labels = labels.to(conf.device)

            # 前向传播：模型预测
            logits = model(input_ids, attention_mask)
            # 计算损失
            loss = loss_fn(logits, labels)
            # 累计损失
            total_loss += loss.item()
            # 获取预测结果（最大logits对应的类别）
            y_pred_list = torch.argmax(logits, dim=1)
            # 存储预测和真实标签，用于计算训练集指标
            train_preds.extend(y_pred_list.cpu().tolist())
            train_labels.extend(labels.cpu().tolist())

            # 梯度清零
            optimizer.zero_grad()
            # 反向传播：计算梯度
            loss.backward()
            # 参数更新：根据梯度更新模型参数
            optimizer.step()

            # 每10个批次或一个轮次结束，计算训练集指标
            if (i + 1) % 10 == 0 or i == len(train_dataloader) - 1:
                # 计算准确率和f1值
                acc = accuracy_score(train_labels, train_preds)
                f1 = f1_score(train_labels, train_preds, average='macro')
                # 获取batch_count，并计算平均损失
                batch_count = i % 10 + 1
                avg_loss = total_loss / batch_count
                # 打印训练信息
                print(f"\n轮次: {epoch + 1}, 批次: {i + 1}, 损失: {avg_loss:.4f}, acc准确率:{acc:.4f}, f1分数:{f1:.4f}")
                # 清空累计损失和预测和真实标签
                total_loss = 0.0
                train_preds, train_labels = [], []

            # 每100个批次或一个轮次结束，计算验证集指标，打印，保存模型
            if (i + 1) % 100 == 0 or i == len(train_dataloader) - 1:
                # 计算在测试集的评估报告，准确率，精确率，召回率，f1值
                report, f1score, accuracy, precision, recall = model2dev(model, dev_dataloader, conf.device)
                print("验证集评估报告：\n", report)
                print(f"验证集的f1: {f1score:.4f}, accuracy:{accuracy:.4f}, precision:{precision:.4f}, recall:{recall:.4f}")
                # 将模型再设置为训练模式
                model.train()
                # 如果验证F1分数优于历史最佳，保存模型
                if f1score > best_f1:
                    # 更新历史最佳F1分数
                    best_f1 = f1score
                    # 保存模型
                    torch.save(model.state_dict(), conf.model_save_path)
                    print("保存模型成功, 当前f1分数:", best_f1)


if __name__ == '__main__':
    model2train()

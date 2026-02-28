#!/usr/bin/env python3
"""
评论数据二次验证脚本
使用 ollama 批量验证字段是否与评论内容相符，防止 LLM 幻觉

优化点（参考 comment_agent.py）：
1. 批量验证 - 一次发送多条评论让模型验证
2. JSON 修复机制 - 处理模型输出的格式错误
3. 补漏机制 - 检测漏输出的条目并重试
4. 并行处理 - 多线程并发
"""

import pandas as pd
import json
import os
import time
import re
import ollama
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# ============== 配置 ==============
# RTX 5070 12GB 配置
MAX_WORKERS = 20      # 并发线程数
BATCH_SIZE = 1       # 每批处理条数
NUM_CTX = 8192       # 上下文长度
MAX_RETRIES = 3      # 最大重试次数

# # 数据路径 (5070 台式机)
# DATA_FILE = r"E:\AIProject\UserProfileGeneration\DataCheck\工作簿5.xlsx"
# OUTPUT_FILE = r"E:\AIProject\UserProfileGeneration\DataCheck\验证结果.xlsx"
# CORRECTED_FILE = r"E:\AIProject\UserProfileGeneration\DataCheck\修正后数据.xlsx"
DATA_FILE = r"/Users/jackjia/Desktop/demo/UserProfileGeneration/DataCheck/工作簿5.xlsx"
OUTPUT_FILE = r"/Users/jackjia/Desktop/demo/UserProfileGeneration/DataCheck/验证结果.xlsx"
CORRECTED_FILE = r"/Users/jackjia/Desktop/demo/UserProfileGeneration/DataCheck/修正后数据.xlsx"
# 使用已安装的模型
# MODEL_NAME = "hf.co/unsloth/Qwen3-4B-GGUF:Q6_K_XL"
OLLAMA_MODEL = None
MODEL_NAME = "hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M"
# 远程 Ollama 服务器地址
OLLAMA_HOST = "http://223.109.141.125:7444"

# 限制验证条数（用于测试，正式运行时设为 None 验证全部）
MAX_VERIFY_COUNT = None  # None = 全部验证

# 是否自动修正错误的字段并保存到Excel
AUTO_CORRECT = True  # True = 自动修正并保存到原文件

# 技术维度
TECH_DIMS = ['智驾', '续航补能', '动力', '驾乘体验', '空间', '外观内饰']
TECH_DIM_DESCS = [dim + '点' for dim in TECH_DIMS]
ALL_FIELDS = ['正负面', '营销手段', '技术亮点'] + TECH_DIMS


def sanitize_comment(comment: str, max_len: int = 500) -> str:
    """净化评论文本，避免特殊字符破坏模型输出的 JSON 结构"""
    text = comment[:max_len] if len(str(comment)) > max_len else str(comment)
    text = text.replace('"', '\u201c').replace('\u201d', '\u201c').replace('"', '\u201c')
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = text.replace('\\', '/')
    return text.strip()


def repair_json(text: str):
    """尝试修复常见 JSON 格式错误后重新解析"""
    # 尝试找到JSON数组
    match = re.search(r'\[[\s\S]*\]', text)
    if not match:
        return None
    raw = match.group()
    # 修复换行问题
    raw = re.sub(r'(?<=: ")(.*?)(?=")', lambda m: m.group().replace('\n', ' ').replace('\r', ' '), raw)
    try:
        return json.loads(raw)
    except Exception:
        pass
    # 尝试逐个解析对象
    results = []
    for obj_str in re.finditer(r'\{[^{}]*\}', raw):
        try:
            results.append(json.loads(obj_str.group()))
        except Exception:
            pass
    return results if results else None


def build_batch_prompt(comments_data):
    """构建批量验证提示词 - 一次验证多条评论"""
    comments_text = []
    for idx, row in comments_data:
        comment = sanitize_comment(row['评论内容'])
        fields_info = f"""- 正负面：{row['正负面']}
- 营销手段：{row['营销手段']}
- 技术亮点：{row['技术亮点']}
- 智驾：{row.get('智驾', '')}，智驾点：{row.get('智驾点', '') if pd.notna(row.get('智驾点', '')) else ''}
- 续航补能：{row.get('续航补能', '')}，续航补能点：{row.get('续航补能点', '') if pd.notna(row.get('续航补能点', '')) else ''}
- 动力：{row.get('动力', '')}，动力点：{row.get('动力点', '') if pd.notna(row.get('动力点', '')) else ''}
- 驾乘体验：{row.get('驾乘体验', '')}，驾乘体验点：{row.get('驾乘体验点', '') if pd.notna(row.get('驾乘体验点', '')) else ''}
- 空间：{row.get('空间', '')}，空间点：{row.get('空间点', '') if pd.notna(row.get('空间点', '')) else ''}
- 外观内饰：{row.get('外观内饰', '')}，外观内饰点：{row.get('外观内饰点', '') if pd.notna(row.get('外观内饰点', '')) else ''}"""

        comments_text.append(f"序号{idx}|评论：{comment}\n现有分类：\n{fields_info}")

    prompt = f"""你是一个数据质量验证专家。请验证以下每条评论的分类是否正确。

判定规则：
1. 严格根据评论原文判断，不添加任何未提及的信息
2. 如果评论没有提及某个维度，该维度应填"无"，对应技术点应为空
3. 如果技术点描述的内容评论中完全没有提到，则为幻觉，应标记为错误
4. "技术亮点"指评论是否主动描述了车辆的技术特性/功能亮点（如智驾功能、续航能力、动力性能等），而非仅提及或抱怨

请对每条评论返回JSON格式的验证结果：
[
{{"序号":序号,"正负面":{{"value":"当前值","correct":true/false,"corrected":"修正值","reason":"理由"}},...}},
...
]

只返回JSON数组，不要其他内容。

评论列表：
"""
    prompt += "\n---\n".join(comments_text)
    return prompt


def make_fail_record(idx):
    """创建失败记录"""
    record = {"序号": idx}
    for field in ALL_FIELDS:
        record[field] = {"value": "", "correct": False, "corrected": "", "reason": "处理失败"}
    return record


def call_model_once(prompt, client):
    """调用模型一次"""
    try:
        response = client.generate(
            model=OLLAMA_MODEL or MODEL_NAME,
            prompt=prompt,
            options={
                "num_ctx": NUM_CTX,
                "temperature": 0.1,
            }
        )
        return response['response'].strip()
    except Exception as e:
        print(f"  模型调用异常: {e}")
        return None


def post_process_results(raw_text):
    """后处理模型输出"""
    if not raw_text:
        return None
    # 清理模型思考标签
    result_text = re.sub(r'[/\\]?no_think[\s\S]*', '', raw_text).strip()
    # 尝试直接解析
    try:
        return json.loads(result_text)
    except json.JSONDecodeError:
        pass
    # 尝试修复JSON
    return repair_json(result_text)


def verify_batch(comments_data, client, max_retries=3):
    """批量验证评论，若漏输出则逐条重试"""
    for attempt in range(max_retries):
        prompt = build_batch_prompt(comments_data)
        raw = call_model_once(prompt, client)

        if raw:
            results = post_process_results(raw)
            if results:
                # 检查是否所有评论都返回了结果
                returned_idxs = {r.get('序号') for r in results if isinstance(r, dict)}
                missing = [(idx, row) for idx, row in comments_data if idx not in returned_idxs]

                if missing:
                    print(f"  [补漏] 批次漏输出 {len(missing)} 条，逐条重试...")
                    for idx, row in missing:
                        single_prompt = build_batch_prompt([(idx, row)])
                        single_raw = call_model_once(single_prompt, client)
                        if single_raw:
                            single_results = post_process_results(single_raw)
                            if single_results:
                                results.extend(single_results)
                            else:
                                results.append(make_fail_record(idx))
                        else:
                            results.append(make_fail_record(idx))
                return results

        print(f"  批次验证失败 (尝试 {attempt+1}/{max_retries})")
        time.sleep(1)

    return [make_fail_record(idx) for idx, _ in comments_data]


def load_and_filter_data():
    """加载并筛选数据"""
    print("=" * 60)
    print("正在加载数据...")
    df = pd.read_excel(DATA_FILE)
    print(f"总数据量: {len(df)}")

    # 筛选条件
    filtered = df[
        (df['有效无效'] == '有效') &
        (df['正负面'].notna() & (df['正负面'] != '')) &
        (df['营销手段'].notna() & (df['营销手段'] != '')) &
        (df['技术亮点'] == '是')
    ].copy()

    print(f"满足筛选条件的数据量: {len(filtered)}")

    if MAX_VERIFY_COUNT:
        filtered = filtered.head(MAX_VERIFY_COUNT)
        print(f"将验证前 {MAX_VERIFY_COUNT} 条数据")

    return filtered


def process_parallel(data, total_workers=8):
    """并行处理验证"""
    total = len(data)
    print(f"\n开始并行验证，共 {total} 条数据")
    print(f"使用模型: {OLLAMA_MODEL or MODEL_NAME}, 并行线程: {total_workers}, 每批: {BATCH_SIZE} 条")
    print(f"远程服务器: {OLLAMA_HOST}")
    print("=" * 60)

    # 测试远程连接
    try:
        test_client = ollama.Client(host=OLLAMA_HOST)
        test_client.list()
        print(f"[OK] 已连接到远程 Ollama 服务器: {OLLAMA_HOST}")
    except Exception as e:
        print(f"[错误] 无法连接到远程 Ollama 服务器: {OLLAMA_HOST}")
        print(f"       错误详情: {e}")
        return [], time.time()

    # 准备数据批次
    data_list = list(data.iterrows())
    batches = []
    for i in range(0, total, BATCH_SIZE):
        batch = data_list[i:i + BATCH_SIZE]
        batches.append(batch)

    print(f"共分为 {len(batches)} 个批次")

    all_results = []
    completed = 0
    failed_batches = []
    start_time = time.time()

    # 初始化 ollama 客户端（连接远程服务器）
    client = ollama.Client(host=OLLAMA_HOST)

    with ThreadPoolExecutor(max_workers=total_workers) as executor:
        future_to_batch = {executor.submit(verify_batch, batch, client): i
                          for i, batch in enumerate(batches)}

        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                results = future.result()
                if results:
                    all_results.extend(results)
                    completed += 1

                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = (len(batches) - completed) / rate if rate > 0 else 0
                    print(f"进度: {completed}/{len(batches)} 批次 | "
                          f"已验证: {len(all_results)} 条 | "
                          f"预计剩余: {remaining/60:.1f}分钟")
                else:
                    failed_batches.append(batch_idx)

            except Exception as e:
                print(f"批次 {batch_idx} 处理异常: {e}")
                failed_batches.append(batch_idx)

    # 重试失败的批次
    if failed_batches:
        print(f"\n重试失败的 {len(failed_batches)} 个批次...")
        for batch_idx in failed_batches:
            results = verify_batch(batches[batch_idx], client, max_retries=5)
            all_results.extend(results)

    return all_results, start_time


def main():
    global DATA_FILE, OUTPUT_FILE, CORRECTED_FILE
    
    print("=" * 60)
    print("评论数据二次验证工具")
    print("=" * 60)
    print(f"自动修正模式: {'开启' if AUTO_CORRECT else '关闭'}")
    print(f"数据文件: {DATA_FILE}")

    # 检查文件是否存在
    if not os.path.exists(DATA_FILE):
        print(f"\n[错误] 数据文件不存在: {DATA_FILE}")
        # 尝试使用备用路径
        alt_path = os.path.join(os.path.dirname(__file__), "工作簿5.xlsx")
        if os.path.exists(alt_path):
            DATA_FILE = alt_path
            OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "验证结果.xlsx")
            CORRECTED_FILE = os.path.join(os.path.dirname(__file__), "修正后数据.xlsx")
            print(f"[切换] 使用本地路径: {DATA_FILE}")
        else:
            return

    # 加载全部原始数据（用于后续修正）
    print("\n正在加载原始数据...")
    original_df = pd.read_excel(DATA_FILE)
    print(f"原始数据总量: {len(original_df)}")

    # 加载需要验证的数据
    data = load_and_filter_data()

    if len(data) == 0:
        print("没有满足条件的数据需要验证")
        return

    # 并行验证
    all_results, start_time = process_parallel(data, MAX_WORKERS)

    # 计算统计
    elapsed = time.time() - start_time
    print(f"\n验证完成! 总耗时: {elapsed:.1f}秒, 平均速度: {len(data)/elapsed:.1f}条/秒")

    # 统计结果
    success_count = sum(1 for r in all_results if isinstance(r, dict) and r.get('正负面', {}).get('correct', False) is not None)
    error_count = len(all_results) - success_count
    print(f"成功: {success_count}, 失败: {error_count}")

    # 分析字段准确率
    print("\n" + "=" * 60)
    print("字段验证结果统计:")
    print("=" * 60)

    fields_to_check = ['正负面', '营销手段', '技术亮点'] + TECH_DIMS

    field_stats = {field: {'correct': 0, 'incorrect': 0} for field in fields_to_check}

    # 记录需要修正的数据
    corrections = []

    for result in all_results:
        if not isinstance(result, dict):
            continue

        for field in fields_to_check:
            if field in result:
                field_data = result[field]
                if isinstance(field_data, dict):
                    if field_data.get('correct', False):
                        field_stats[field]['correct'] += 1
                    else:
                        field_stats[field]['incorrect'] += 1
                        corrected_value = field_data.get('corrected', '')
                        if corrected_value:
                            corrections.append((result.get('序号'), field, corrected_value))

    for field, stats in field_stats.items():
        total_check = stats['correct'] + stats['incorrect']
        if total_check > 0:
            accuracy = stats['correct'] / total_check * 100
            print(f"  {field}: 正确 {stats['correct']}, 错误 {stats['incorrect']}, 准确率 {accuracy:.1f}%")
        else:
            print(f"  {field}: 无数据")

    # 自动修正并保存
    if AUTO_CORRECT and corrections:
        print("\n" + "=" * 60)
        print("正在进行自动修正...")
        print("=" * 60)

        corrected_df = original_df.copy()
        correction_count = 0

        for seq_no, field, new_value in corrections:
            if seq_no is None:
                continue
            mask = corrected_df['序号'] == seq_no
            if mask.any():
                corrected_df.loc[mask, field] = new_value
                correction_count += 1
                if correction_count <= 10:
                    print(f"  序号 {seq_no}: {field} -> {new_value}")

        if correction_count > 10:
            print(f"  ... 还有 {correction_count - 10} 处修正")

        # 保存修正后的数据
        corrected_df.to_excel(DATA_FILE, index=False)
        print(f"\n修正完成! 共修正 {correction_count} 处错误")
        print(f"已更新原文件: {DATA_FILE}")

        corrected_df.to_excel(CORRECTED_FILE, index=False)
        print(f"备份保存到: {CORRECTED_FILE}")
    elif AUTO_CORRECT:
        print("\n没有发现需要修正的错误")

    # 保存详细验证结果到Excel
    print("\n正在保存验证结果...")

    output_rows = []
    for result in all_results:
        if not isinstance(result, dict):
            continue

        seq_no = result.get('序号')
        # 找到原始评论内容
        original_row = data[data['序号'] == seq_no]
        if original_row.empty:
            continue

        row_data = {
            '序号': seq_no,
            '评论内容': original_row.iloc[0]['评论内容'],
            '状态': '成功' if any(result.get(f, {}).get('correct', True) for f in fields_to_check) else '失败',
        }

        for field in fields_to_check:
            if field in result:
                field_data = result[field]
                if isinstance(field_data, dict):
                    is_correct = field_data.get('correct', False)
                    row_data[f'{field}_验证'] = '正确' if is_correct else '错误'
                    row_data[f'{field}_理由'] = field_data.get('reason', '')
                    corrected = field_data.get('corrected', '')
                    if corrected and not is_correct:
                        row_data[f'{field}_修正为'] = corrected
                    row_data[f'{field}_原值'] = field_data.get('value', '')
                else:
                    row_data[f'{field}_验证'] = '未知'
            else:
                row_data[f'{field}_验证'] = '未知'

        row_data['原始JSON'] = json.dumps(result, ensure_ascii=False)
        output_rows.append(row_data)

    output_df = pd.DataFrame(output_rows)
    output_df.to_excel(OUTPUT_FILE, index=False)
    print(f"验证结果已保存到: {OUTPUT_FILE}")

    # 打印错误示例
    print("\n" + "=" * 60)
    print("发现的问题示例（仅显示前5条）:")
    print("=" * 60)

    error_examples = [r for r in all_results[:20] if isinstance(r, dict) and any(
        not r.get(f, {}).get('correct', True) for f in fields_to_check
    )][:5]

    for i, example in enumerate(error_examples):
        seq_no = example.get('序号')
        original_row = data[data['序号'] == seq_no]
        comment = original_row.iloc[0]['评论内容'] if not original_row.empty else ''

        print(f"\n--- 示例 {i + 1} (序号: {seq_no}) ---")
        print(f"评论: {str(comment)[:100]}...")

        for field in fields_to_check:
            if field in example:
                field_data = example[field]
                if isinstance(field_data, dict) and not field_data.get('correct', True):
                    corrected = field_data.get('corrected', '无')
                    reason = field_data.get('reason', '')[:50]
                    original_val = field_data.get('value', '')
                    print(f"  {field}: 原值={original_val} -> 修正为={corrected}")
                    print(f"    理由: {reason}")

    print("\n" + "=" * 60)
    print("验证完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

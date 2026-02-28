#!/usr/bin/env python3
"""
评论数据二次验证脚本
使用 ollama 验证字段是否与评论内容相符，防止 LLM 幻觉

筛选条件：
- 有效无效 = 有效
- 正负面 非空
- 营销手段 非空
- 技术亮点 = 是

验证内容：
- 正负面、营销手段、技术亮点 是否正确
- 各技术维度（智驾/续航补能/动力/驾乘体验/空间/外观内饰）是否正确
- 技术点描述是否与评论内容相符（不是幻觉）
"""

import pandas as pd
import json
import os
import time
import ollama
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============== 配置 ==============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "工作簿5.xlsx")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "验证结果.xlsx")

# 使用已安装的模型
MODEL_NAME = "hf.co/unsloth/Qwen3-4B-GGUF:Q6_K_XL"

# 性能配置
MAX_WORKERS = 4  # 并发数
BATCH_SIZE = 5   # 每批处理条数
NUM_CTX = 8192

# 限制验证条数（用于测试，正式运行时设为 None 验证全部）
MAX_VERIFY_COUNT = None  # None = 全部验证

# 技术维度
TECH_DIMS = ['智驾', '续航补能', '动力', '驾乘体验', '空间', '外观内饰']
TECH_DIM_DESCS = [dim + '点' for dim in TECH_DIMS]

# ============== 提示词模板 ==============
VERIFY_PROMPT = """你是一个数据质量验证专家。请根据评论内容验证以下分类是否正确。

评论内容：
{comment}

现有分类：
- 正负面：{sentiment}
- 营销手段：{marketing}
- 技术亮点：{tech_highlight}
- 智驾：{zhijia}，智驾点：{zhijia_point}
- 续航补能：{xuhang}，续航补能点：{xuhang_point}
- 动力：{dongli}，动力点：{dongli_point}
- 驾乘体验：{jiaju}，驾乘体验点：{jiaju_point}
- 空间：{kongjian}，空间点：{kongjian_point}
- 外观内饰：{neishi}，外观内饰点：{neishi_point}

请判断每个字段是否与评论内容相符。如果不相符或为幻觉，请给出正确的分类。

判定规则：
1. 严格根据评论原文判断，不添加任何未提及的信息
2. 如果评论没有提及某个维度，该维度应填"无"，对应技术点应为空
3. 如果技术点描述的内容评论中完全没有提到，则为幻觉，应标记为错误

请返回JSON格式的验证结果：
{{
    "正负面": {{"value": "当前值", "correct": true/false, "reason": "判断理由"}},
    "营销手段": {{"value": "当前值", "correct": true/false, "reason": "判断理由"}},
    "技术亮点": {{"value": "当前值", "correct": true/false, "reason": "判断理由"}},
    "智驾": {{"value": "当前值", "correct": true/false, "reason": "判断理由", "point_correct": true/false}},
    "续航补能": {{"value": "当前值", "correct": true/false, "reason": "判断理由", "point_correct": true/false}},
    "动力": {{"value": "当前值", "correct": true/false, "reason": "判断理由", "point_correct": true/false}},
    "驾乘体验": {{"value": "当前值", "correct": true/false, "reason": "判断理由", "point_correct": true/false}},
    "空间": {{"value": "当前值", "correct": true/false, "reason": "判断理由", "point_correct": true/false}},
    "外观内饰": {{"value": "当前值", "correct": true/false, "reason": "判断理由", "point_correct": true/false}}
}}

只返回JSON，不要其他内容。"""


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


def build_prompt(row):
    """构建验证提示词"""
    comment = row['评论内容']
    return VERIFY_PROMPT.format(
        comment=comment[:500] if len(str(comment)) > 500 else comment,
        sentiment=row['正负面'],
        marketing=row['营销手段'],
        tech_highlight=row['技术亮点'],
        zhijia=row.get('智驾', ''),
        zhijia_point=row.get('智驾点', '') if pd.notna(row.get('智驾点', '')) else '',
        xuhang=row.get('续航补能', ''),
        xuhang_point=row.get('续航补能点', '') if pd.notna(row.get('续航补能点', '')) else '',
        dongli=row.get('动力', ''),
        dongli_point=row.get('动力点', '') if pd.notna(row.get('动力点', '')) else '',
        jiaju=row.get('驾乘体验', ''),
        jiaju_point=row.get('驾乘体验点', '') if pd.notna(row.get('驾乘体验点', '')) else '',
        kongjian=row.get('空间', ''),
        kongjian_point=row.get('空间点', '') if pd.notna(row.get('空间点', '')) else '',
        neishi=row.get('外观内饰', ''),
        neishi_point=row.get('外观内饰点', '') if pd.notna(row.get('外观内饰点', '')) else '',
    )


def verify_single(row, client):
    """验证单条数据"""
    try:
        prompt = build_prompt(row)

        response = client.generate(
            model=MODEL_NAME,
            prompt=prompt,
            options={
                "num_ctx": NUM_CTX,
                "temperature": 0.1,
            }
        )

        # 解析JSON结果
        result_text = response['response'].strip()

        # 尝试提取JSON
        try:
            # 尝试直接解析
            result = json.loads(result_text)
        except json.JSONDecodeError:
            # 尝试从文本中提取JSON
            import re
            json_match = re.search(r'\{[\s\S]*\}', result_text)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = {"error": "无法解析结果", "raw": result_text}

        return {
            '序号': row['序号'],
            '评论内容': row['评论内容'],
            '验证结果': result,
            '状态': '成功'
        }

    except Exception as e:
        return {
            '序号': row['序号'],
            '评论内容': row['评论内容'],
            '验证结果': {'error': str(e)},
            '状态': '失败'
        }


def analyze_batch(rows, client):
    """批量分析"""
    results = []
    for _, row in rows.iterrows():
        result = verify_single(row, client)
        results.append(result)
    return results


def main():
    print("=" * 60)
    print("评论数据二次验证工具")
    print("=" * 60)

    # 加载数据
    data = load_and_filter_data()

    if len(data) == 0:
        print("没有满足条件的数据需要验证")
        return

    # 初始化 ollama 客户端
    print(f"\n使用模型: {MODEL_NAME}")
    client = ollama.Client()

    # 验证每条数据
    print("\n开始验证...")
    all_results = []
    total = len(data)

    start_time = time.time()

    # 逐条处理（更稳定）
    for idx, (_, row) in enumerate(data.iterrows()):
        if (idx + 1) % 50 == 0 or idx == 0:
            elapsed = time.time() - start_time
            speed = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (total - idx - 1) / speed if speed > 0 else 0
            print(f"进度: {idx + 1}/{total} ({100 * (idx + 1) / total:.1f}%) | 速度: {speed:.1f}条/秒 | 预计剩余: {eta:.0f}秒")

        result = verify_single(row, client)
        all_results.append(result)

        # 每次请求间隔一下，避免并发过高
        time.sleep(0.1)

    elapsed = time.time() - start_time
    print(f"\n验证完成! 总耗时: {elapsed:.1f}秒, 平均速度: {total/elapsed:.1f}条/秒")

    # 统计结果
    success_count = sum(1 for r in all_results if r['状态'] == '成功')
    error_count = sum(1 for r in all_results if r['状态'] == '失败')
    print(f"成功: {success_count}, 失败: {error_count}")

    # 分析字段准确率
    print("\n" + "=" * 60)
    print("字段验证结果统计:")
    print("=" * 60)

    fields_to_check = ['正负面', '营销手段', '技术亮点'] + TECH_DIMS

    field_stats = {field: {'correct': 0, 'incorrect': 0, 'error': 0} for field in fields_to_check}

    for result in all_results:
        if result['状态'] != '成功':
            continue

        verify_result = result['验证结果']
        if 'error' in verify_result:
            continue

        for field in fields_to_check:
            if field in verify_result:
                if verify_result[field].get('correct', False):
                    field_stats[field]['correct'] += 1
                else:
                    field_stats[field]['incorrect'] += 1
            else:
                field_stats[field]['error'] += 1

    for field, stats in field_stats.items():
        total_check = stats['correct'] + stats['incorrect']
        if total_check > 0:
            accuracy = stats['correct'] / total_check * 100
            print(f"  {field}: 正确 {stats['correct']}, 错误 {stats['incorrect']}, 准确率 {accuracy:.1f}%")
        else:
            print(f"  {field}: 无数据")

    # 保存详细结果到Excel
    print("\n正在保存结果...")

    # 整理输出数据
    output_rows = []
    for result in all_results:
        row_data = {
            '序号': result['序号'],
            '评论内容': result['评论内容'],
            '状态': result['状态'],
        }

        if result['状态'] == '成功':
            verify_result = result['验证结果']

            # 原有字段
            for field in fields_to_check:
                if field in verify_result:
                    row_data[f'{field}_验证'] = '正确' if verify_result[field].get('correct') else '错误'
                    row_data[f'{field}_理由'] = verify_result[field].get('reason', '')
                else:
                    row_data[f'{field}_验证'] = '未知'
                    row_data[f'{field}_理由'] = ''

            # 原始值
            for field in fields_to_check:
                if field in verify_result:
                    row_data[f'{field}_原值'] = verify_result[field].get('value', '')

            row_data['原始JSON'] = json.dumps(verify_result, ensure_ascii=False)
        else:
            for field in fields_to_check:
                row_data[f'{field}_验证'] = '失败'
            row_data['原始JSON'] = str(result['验证Result'])

        output_rows.append(row_data)

    output_df = pd.DataFrame(output_rows)
    output_df.to_excel(OUTPUT_FILE, index=False)
    print(f"结果已保存到: {OUTPUT_FILE}")

    # 打印错误示例
    print("\n" + "=" * 60)
    print("发现的问题示例（仅显示前5条）:")
    print("=" * 60)

    error_examples = []
    for result in all_results:
        if result['状态'] != '成功':
            continue

        verify_result = result['验证结果']
        if 'error' in verify_result:
            continue

        has_error = False
        for field in fields_to_check:
            if field in verify_result and not verify_result[field].get('correct', True):
                has_error = True
                break

        if has_error:
            error_examples.append(result)
            if len(error_examples) >= 5:
                break

    for i, example in enumerate(error_examples):
        print(f"\n--- 示例 {i + 1} (序号: {example['序号']}) ---")
        print(f"评论: {example['评论内容'][:100]}...")

        verify_result = example['验证Result']
        for field in fields_to_check:
            if field in verify_result:
                if not verify_result[field].get('correct', True):
                    print(f"  {field}: 原值={verify_result[field].get('value', '')}, 错误: {verify_result[field].get('reason', '')[:50]}")


if __name__ == "__main__":
    main()

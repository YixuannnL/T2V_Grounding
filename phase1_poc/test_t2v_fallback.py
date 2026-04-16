#!/usr/bin/env python3
"""
测试 T2V fallback 逻辑：
- 当只有 location ref 而没有 subject（character/object）ref 时，应回退 T2V
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from entity_parser.parser import Entity


def test_generation_mode_decision():
    """模拟 pipeline 中的 generation_mode 决策逻辑"""

    # 测试场景 1: 有 subject ref → 应该用 S2V
    print("=" * 60)
    print("场景 1: 有 character ref")
    non_location_entities = [
        Entity(entity_id="char_alex", type="character", text_description="Alex",
               attributes={}, is_new=False, grounding_priority="high"),
    ]
    location_entities = [
        Entity(entity_id="loc_office", type="location", text_description="Office",
               attributes={}, is_new=False, grounding_priority="medium"),
    ]
    reference_used = {"char_alex": ["path/to/alex.jpg"], "loc_office": ["path/to/office.jpg"]}

    has_subject_ref = any(e.entity_id in reference_used for e in non_location_entities)
    generation_mode = "phantom" if has_subject_ref else "t2v"
    print(f"  has_subject_ref: {has_subject_ref}")
    print(f"  generation_mode: {generation_mode}")
    assert generation_mode == "phantom", "应该使用 S2V (phantom)"
    print("  ✓ 通过\n")

    # 测试场景 2: 只有 location ref → 应该回退 T2V
    print("=" * 60)
    print("场景 2: 只有 location ref，无 subject ref")
    non_location_entities = [
        Entity(entity_id="char_bob", type="character", text_description="Bob",
               attributes={}, is_new=True, grounding_priority="high"),
    ]
    location_entities = [
        Entity(entity_id="loc_office", type="location", text_description="Office",
               attributes={"lighting": "fluorescent", "atmosphere": "busy"},
               is_new=False, grounding_priority="medium"),
    ]
    reference_used = {"loc_office": ["path/to/office.jpg"]}  # 只有 location

    has_subject_ref = any(e.entity_id in reference_used for e in non_location_entities)
    num_location_refs = len([e for e in location_entities if e.entity_id in reference_used])

    if has_subject_ref:
        generation_mode = "phantom"
    else:
        generation_mode = "t2v"
        if num_location_refs > 0:
            print(f"  ⚠️  无 subject 参考图，仅有 {num_location_refs} 个 location ref，回退 T2V")
            # 构建环境描述
            env_parts = []
            for loc in location_entities:
                if loc.entity_id in reference_used:
                    desc = loc.text_description
                    if loc.attributes:
                        attrs = ", ".join(f"{k}: {v}" for k, v in loc.attributes.items())
                        desc = f"{desc} ({attrs})"
                    env_parts.append(f"Scene: {desc}.")
            if env_parts:
                env_context = "[Environment Context]\n" + "\n".join(env_parts)
                print(f"  环境描述注入 prompt:\n{env_context}")

    print(f"  has_subject_ref: {has_subject_ref}")
    print(f"  generation_mode: {generation_mode}")
    assert generation_mode == "t2v", "应该回退 T2V"
    print("  ✓ 通过\n")

    # 测试场景 3: 无任何 ref → T2V（首镜头）
    print("=" * 60)
    print("场景 3: 无任何 ref（首镜头）")
    reference_used = {}

    has_subject_ref = any(e.entity_id in reference_used for e in non_location_entities)
    generation_mode = "phantom" if has_subject_ref else "t2v"
    print(f"  has_subject_ref: {has_subject_ref}")
    print(f"  generation_mode: {generation_mode}")
    assert generation_mode == "t2v", "应该使用 T2V"
    print("  ✓ 通过\n")

    print("=" * 60)
    print("所有测试通过!")


if __name__ == "__main__":
    test_generation_mode_decision()

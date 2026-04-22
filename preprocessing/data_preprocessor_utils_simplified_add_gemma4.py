import os
import json
import yaml
import random
import re
from typing import List, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum
from collections import Counter
from tqdm.auto import tqdm

#################################################################################################
# CONSTANTS & NORMALIZERS
#################################################################################################
NEGATIVE_VARIANTS = {"no_relation", "no_relationship", "no relationship"}

# BEE 속성 영문 → 한글 매핑 (사용자 정책: BEE 속성은 한글로 통일)
BEE_ENG_TO_KOR = {
    "Loyalty": "충성도", "Effect": "효과", "Feel": "사용감",
    "Moisturizing Power": "보습력", "Convenience": "편리성", "Color": "색상",
    "Spreadability": "발림성", "Cleansing Power": "세정력", "Texture": "제형",
    "Longevity": "지속력", "Perceived Price": "인지가격", "Scent": "향",
    "Coverage": "커버력", "Capacity": "용량", "Package/Container Design": "패키지/용기_디자인",
    "Side Effects/Damage": "부작용/손상", "Ingredients": "성분", "Versatility": "활용성",
    "Single Item Design": "단품_디자인", "Pigmented/Color Payoff": "발색력",
    "Adhesion": "밀착력", "Absorption": "흡수력", "Quality": "품질",
    "Composition": "구성", "Delivery": "배송", "Expressiveness": "표현력",
    "Glossiness": "광감", "Portability": "휴대성", "Taste": "맛",
    "Promotion": "판촉", "Curling/Volume": "컬링/볼륨", "White Cast": "백탁현상",
    "Smudging": "번짐", "Clumping": "뭉침", "Shelf Life": "유통기한",
    "Powder Fallout": "가루날림", "Pearl Effect": "펄감", "Breakdown": "무너짐",
    "Service": "서비스",
}
BEE_LABELS_ALL = set(BEE_ENG_TO_KOR.keys()) | set(BEE_ENG_TO_KOR.values())

# NER-NER 페어에 BEE 속성 라벨이 잘못 들어간 경우의 매핑 (ner_ner_relation_list와 정합)
BEE_LABEL_TO_NER_NER_MAP = {
    "Ingredients": "has_ingredient",
    "Effect": "affects",
    "Composition": "has_part",
    "Capacity": "has_attribute",
}

def normalize_negative_label(rel):
    """negative 라벨 이형태(no_relationship, NO_RELATIONSHIP 등)를 'NO_RELATION'으로 통일."""
    if rel is None:
        return "NO_RELATION"
    if rel.strip().lower().replace(" ", "_") in NEGATIVE_VARIANTS:
        return "NO_RELATION"
    return rel

def normalize_bee_label(rel):
    """BEE 속성 영문 라벨을 한글로 통일. BEE 속성이 아닌 라벨은 그대로 반환."""
    if rel is None:
        return rel
    return BEE_ENG_TO_KOR.get(rel, rel)

#################################################################################################
# ENUMS & CONFIGS
#################################################################################################
# Mode 선택: 데이터 생성 테스크 설정
class DataGenerationTask(Enum):
    NER_NER = "ner_ner" # NER-NER 관계 추출
    NER_BEE = "ner_bee" # NER-BEE 관계 추출 (Relation = False도 포함)
    NER_BEE_TRUE_ONLY = "ner_bee_true_only" # NER-BEE 관계 추출 (Relation = True만 포함)

# Mode Configuration: 학습 데이터/Prompt 생성 모드/구성을 설정하는 데이터 클래스
@dataclass
class DataGenerationModeConfig:
    enable_description: bool = False # 프롬프트에 Description을 포함할지 여부
    summarize_description: bool = False # Description을 요약본/원본으로 제공 할지 여부
    few_shot: bool = False # Few-shot 예재를 프롬프트에 포함 할지 여부
    selected_few_shot: bool = False # Random이 아닌 Selected Fewshot을 사용할지 여부
    reasoning: bool = False # 모델의 추론 (Reasoning) 과정을 요구할지 여부

    def __post_init__(self):
        # Conflict 방지: 설명 비활성화인데 요약이 켜져있으면 요약 끄기
        if not self.enable_description and self.summarize_description:
            self.summarize_description = False
        # Conflict 방지: Fewshot이 비활성화인데 Selected_Fewshot이 켜져있으면 끄기
        if not self.few_shot and self.selected_few_shot:
            self.selected_few_shot = False

#################################################################################################
# DATA PREPROCESSOR
#################################################################################################
class DataPreprocessor:
    """
    설명: raw_data에서 학습 데이터를 구축하기 위한 전처리 클래스 
    """
    def __init__(self):
        # 영문 Relation명을 한글로 매핑하기 위한 dictionary. 
        self.ENG_TO_KOR_MAP = {
            "Loyalty": "충성도", "Effect": "효과", "Feel": "사용감",
            "Moisturizing Power": "보습력", "Convenience": "편리성", "Color": "색상",
            "Spreadability": "발림성", "Cleansing Power": "세정력", "Texture": "제형",
            "Longevity": "지속력", "Perceived Price": "인지가격", "Scent": "향",
            "Coverage": "커버력", "Capacity": "용량", "Package/Container Design": "패키지/용기_디자인",
            "Side Effects/Damage": "부작용/손상", "Ingredients": "성분", "Versatility": "활용성",
            "Single Item Design": "단품_디자인", "Pigmented/Color Payoff": "발색력",
            "Adhesion": "밀착력", "Absorption": "흡수력", "Quality": "품질",
            "Composition": "구성", "Delivery": "배송", "Expressiveness": "표현력",
            "Glossiness": "광감", "Portability": "휴대성", "Taste": "맛",
            "Promotion": "판촉", "Curling/Volume": "컬링/볼륨", "White Cast": "백탁현상",
            "Smudging": "번짐", "Clumping": "뭉침", "Shelf Life": "유통기한",
            "Powder Fallout": "가루날림", "Pearl Effect": "펄감", "Breakdown": "무너짐", "Service": "서비스",
            "NO_RELATION": "NO_RELATION", "used_by": "사용자", "used_on": "사용부위", "same_entity": "동일_개체", 
            "applied_to": "적용_대상"
        }

    def remove_negative_relations(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Input: Raw Data: List[Dict]
        Output: "NO_RELATION"이 제거된 List[Dict]
        설명: BERT에서 추론한 결과인 Gold_Label(정답)에서 "NO_RELATION"이 포함된 후보 문자열 필터링 
        """
        out_data = []
        for doc in data:
            new_cands = []
            new_meta = []
            
            # 메타데이터를 pair_id 기준으로 매핑하여 빠른 조회 가능하게 설정
            meta_dict = {m["pair_id"]: m for m in doc.get("meta_info", [])}

            for cand in doc.get("candidate_pairs", []):
                if normalize_negative_label(cand.get("relation")) != "NO_RELATION":
                    new_cands.append(cand)
                    if cand.get("pair_id") in meta_dict:
                        new_meta.append(meta_dict[cand["pair_id"]])
            
            if new_cands: # Only keep the doc if surviving relations exist
                new_doc = doc.copy()
                new_doc["candidate_pairs"] = new_cands
                new_doc["meta_info"] = new_meta
                out_data.append(new_doc)
        return out_data

    def extract_and_classify(self, data):
        """
        Input: 전처리된 문서 리스트
        Output: Task별로 분류된 Dict (ner_ner / ner_bee / failed)
        설명:
        - source_type에 따라 Task 버킷으로 분류
        - BEE-BEE 페어는 drop (BEE-BEE Task 미사용)
        - BEE-NER 페어는 NER-BEE 버킷에 합치되 relation = NO_RELATION으로 강제 (방향성 오류 → negative 학습)
        - NER-BEE 페어는 BEE 속성 영→한 통일 + negative 정규화
        - NER-NER 페어 안 BEE 라벨 노이즈는 매핑 가능시 매핑, 불가시 drop
        """
        ner_ner, ner_bee, failed = [], [], []
        # 통계
        stats = {"ner_nr_bee_label_mapped": 0, "ner_nr_bee_label_dropped": 0, "bee_ner_forced_neg": 0, "bee_bee_dropped": 0}

        for doc in data:
            if not doc.get("candidate_pairs"):
                failed.append(doc)
                continue

            buckets = {
                "NER-NER": {"c": [], "m": []},
                "NER-BEE": {"c": [], "m": []},
            }

            meta_dict = {m["pair_id"]: m for m in doc.get("meta_info", [])}

            for cand in doc.get("candidate_pairs", []):
                pair_id = cand.get("pair_id")
                meta = meta_dict.get(pair_id, {})

                s_type = meta.get("subject", {}).get("source_type", "NER")
                o_type = meta.get("object", {}).get("source_type", "NER")
                rel_source = f"{s_type}-{o_type}".upper()

                if rel_source == "BEE-BEE":
                    stats["bee_bee_dropped"] += 1
                    continue

                if rel_source == "BEE-NER":
                    # 방향성 오류 → NO_RELATION으로 강제하고 NER-BEE 버킷에 합침
                    cand = {**cand, "relation": "NO_RELATION"}
                    stats["bee_ner_forced_neg"] += 1
                    target_bucket = "NER-BEE"
                elif rel_source == "NER-BEE":
                    # BEE 속성 영→한 통일 + negative 정규화
                    raw_rel = cand.get("relation")
                    rel = normalize_negative_label(raw_rel)
                    if rel != "NO_RELATION":
                        rel = normalize_bee_label(rel)
                    cand = {**cand, "relation": rel}
                    target_bucket = "NER-BEE"
                elif rel_source == "NER-NER":
                    raw_rel = cand.get("relation")
                    rel = normalize_negative_label(raw_rel)
                    if rel in BEE_LABEL_TO_NER_NER_MAP:
                        rel = BEE_LABEL_TO_NER_NER_MAP[rel]
                        stats["ner_nr_bee_label_mapped"] += 1
                    elif rel in BEE_LABELS_ALL:
                        # 매핑 불가한 BEE 라벨 → drop
                        stats["ner_nr_bee_label_dropped"] += 1
                        continue
                    cand = {**cand, "relation": rel}
                    target_bucket = "NER-NER"
                else:
                    # 알 수 없는 source_type → drop
                    continue

                buckets[target_bucket]["c"].append(cand)
                if meta:
                    buckets[target_bucket]["m"].append(meta)

            def make_doc(b_type):
                if buckets[b_type]["c"]:
                    nd = doc.copy()
                    nd["candidate_pairs"] = buckets[b_type]["c"]
                    nd["meta_info"] = buckets[b_type]["m"]
                    return nd
                return None

            if nn_doc := make_doc("NER-NER"): ner_ner.append(nn_doc)
            if nb_doc := make_doc("NER-BEE"): ner_bee.append(nb_doc)

        # 통계 출력
        print(f"📊 extract_and_classify stats: {stats}")
        return {"ner_ner": ner_ner, "ner_bee": ner_bee, "failed": failed, "_stats": stats}

#################################################################################################
# CHUNKER
#################################################################################################
class RelationBasedChunker:
    """
    설명: sLLM maxlen 컨텍스트 길이 제한을 보호하기 위해, candidate_pair 개수 기준으로 문서를 chunking 하는 클래스.
    """
    _TAG_RE = re.compile(r'\[([A-Z_][A-Z_0-9/]*\d+)\]')

    @classmethod
    def _extract_tag_ids(cls, text_like: str):
        """Extract entity tag identifiers (e.g., 'PER1', 'MOISTURIZING_POWER2') from a string."""
        return set(cls._TAG_RE.findall(text_like or ""))

    @classmethod
    def _narrow_text_to_chunk(cls, text: str, chunk_cands, padding: int = 200) -> str:
        """
        Return the minimal substring of `text` that still contains every entity tag
        referenced by the chunk's candidate pairs, plus a little padding on each side.

        If any referenced tag isn't found in `text`, fall back to returning `text`
        unchanged (safer than silently dropping context the model needs).
        """
        if not text or not chunk_cands:
            return text
        needed = set()
        for c in chunk_cands:
            needed |= cls._extract_tag_ids(str(c.get("subject", "")))
            needed |= cls._extract_tag_ids(str(c.get("object", "")))
        if not needed:
            return text

        positions = []
        for tag in needed:
            marker = f"[{tag}]"
            idx = text.find(marker)
            if idx < 0:
                return text  # fallback: tag not locatable → keep original
            positions.append(idx)
            positions.append(idx + len(marker))
            close_marker = f"[/{tag}]"
            close_idx = text.find(close_marker)
            if close_idx >= 0:
                positions.append(close_idx + len(close_marker))
        start = max(0, min(positions) - padding)
        end = min(len(text), max(positions) + padding)
        # Only narrow if we actually save meaningful characters (>30% reduction).
        if (end - start) >= len(text) * 0.7:
            return text
        return text[start:end]

    def chunk_by_relations(self, entry: Dict[str, Any], target_task=None, entity_threshold: int = 20) -> List[Dict[str, Any]]:
        """
        Input: 단일 문서 Dict (entry), target_task, entity_threshold
        Output: entity_threshold에 맞춰 여러 개로 쪼개진 문서 List[Dict]
        설명: 하나의 문서 내 candidate_pair이 너무 많으면 entity_threshold 단위로 잘라서 새 문서를 여러개 만드는 함수.
              청크당 text는 해당 청크의 pair들이 참조하는 entity tag를 모두 포함하는
              최소 substring(+padding)으로 축소해 pair-dense 문서의 text 복제를 완화.
        """
        cands = entry.get("candidate_pairs", [])
        metas = entry.get("meta_info", [])
        text = entry.get("text", "")

        # entity_threshold 는 청크당 MAX candidate_pair 개수로 해석 됨. 제한을 넘지 않으면 원본을 그대로 반환
        if len(cands) <= entity_threshold:
            return [entry]

        # entity_threshold 단위로 슬라이싱하여 청크 생성
        chunks = []
        for i in range(0, len(cands), entity_threshold):
            chunk_cands = cands[i:i+entity_threshold]
            chunk_entry = entry.copy()
            chunk_entry["id"] = f"{entry['id']}_chunk_{i//entity_threshold}"
            chunk_entry["origin_id"] = entry.get("origin_id", entry["id"])
            # Narrow text to the region covering this chunk's tags (safe fallback to full text)
            chunk_entry["text"] = self._narrow_text_to_chunk(text, chunk_cands)

            chunk_entry["candidate_pairs"] = chunk_cands
            if metas:
                chunk_entry["meta_info"] = metas[i:i+entity_threshold]

            chunks.append(chunk_entry)

        return chunks

#################################################################################################
# PREPROCESS INPUTS (STRICT ORDER & GATEKEEPER)
#################################################################################################
class PreprocessInput:
    """
    설명: 모델에 들어갈 최종 Input 문자열 (Tag 포함)과 Output 문자열 (정답 JSON) 문자열을 조립합니다.
    """
    def __init__(self, text_tag="TEXT", cand_tag="CANDIDATE_PAIRS"):
        self.text_tag = text_tag
        self.cand_tag = cand_tag

    def build_input_output(self, entries, task_type):
        results = []
        
        # 🎯 MEMORY APPLIED: Use "is_relational" for NER-BEE tasks!
        rel_key = "is_relational" if task_type in [DataGenerationTask.NER_BEE, DataGenerationTask.NER_BEE_TRUE_ONLY] else "relation"

        for e in entries:
            text = e.get("text", "")
            cands = e.get("candidate_pairs", [])

            clean_input_cands = []
            output_list = []
            
            for cand in cands:
                # Fallback safeguard
                if isinstance(cand, str):
                    cand = {"subject": "UNKNOWN", "object": "UNKNOWN", "relation": "NO_RELATION", "pair_id": "UNK"}
                    
                sub_str = str(cand.get("subject", "UNKNOWN")).replace('"', '')
                obj_str = str(cand.get("object", "UNKNOWN")).replace('"', '')
                
                # 🎯 NEW: Aggressively strip whitespace between the tags and the text!
                # Turns "[PER1] Reviewer [/PER1]" into "[PER1]Reviewer[/PER1]"
                sub_str = re.sub(r'(\[[^\]]+\])\s*(.*?)\s*(\[/[^\]]+\])', r'\1\2\3', sub_str).strip()
                obj_str = re.sub(r'(\[[^\]]+\])\s*(.*?)\s*(\[/[^\]]+\])', r'\1\2\3', obj_str).strip()
                
                raw_rel = normalize_negative_label(cand.get("relation", "NO_RELATION"))
                pair_id = cand.get('pair_id', 'UNK')
                formatted_pair_id = f"[{pair_id}]"

                # NER-BEE / NER-BEE_TRUE_ONLY는 is_relational 키가 true/false binary여야 하므로
                # NO_RELATION은 "false", 그 외 속성/관계명 라벨은 모두 "true"로 정규화.
                # (system_prompt_ner_bee와 output_format_w_keys_ner_bee의 contract와 일치시킴)
                if task_type in [DataGenerationTask.NER_BEE, DataGenerationTask.NER_BEE_TRUE_ONLY]:
                    output_rel = "false" if raw_rel in (None, "NO_RELATION") else "true"
                else:
                    output_rel = raw_rel

                # 1. 🎯 INPUT: Format the candidate pair as a JSON dictionary (null placeholder)
                input_cand_dict = {
                    "pair_id": formatted_pair_id,
                    "subject": sub_str,
                    rel_key: None,  # 🎯 This automatically becomes `null` when json.dumps() runs!
                    "object": obj_str
                }
                clean_input_cands.append(input_cand_dict)

                # 2. 🎯 OUTPUT: Same JSON structure, but including the actual relation
                strict_order_dict = {
                    "pair_id": formatted_pair_id,
                    "subject": sub_str,
                    rel_key: output_rel,
                    "object": obj_str
                }
                output_list.append(strict_order_dict)

            # JSON Input 구조화 
            input_dict = {
                "text": text,
                "candidate_pairs": clean_input_cands
            }

            # 3. 🎯 Dump both Input and Output with indent=2 so they look perfectly mirrored
            final_json_input = json.dumps(input_dict, indent=2, ensure_ascii=False)
            final_json_output = json.dumps(output_list, indent=2, ensure_ascii=False)
            
            results.append({
                "id": e.get("id"), 
                "text": text,
                "candidate_pairs": clean_input_cands,
                "input": final_json_input,
                "output": final_json_output
            })
            
        return results

#################################################################################################
# PROMPT COMPILER
#################################################################################################
#################################################################################################
# PROMPT COMPILER (UPDATED FOR GEMMA 4 + BEE-BEE FIX)
#################################################################################################
class PromptCompiler:
    """
    설명: YAML Template에 있는 Prompt들을 기반으로 system, user, model의 input/output를 합치고 선택된 모델의 Tokenizer 형식에 맞춰 최종 문자열을 반환합니다. 
    """
    def __init__(self, task, mode_config, template_yaml_path: str, raw_data_sources: List[dict], seed: int = 42, model_name: str = None, tokenizer = None):
        self.task = task
        self.mode_config = mode_config
        self.tokenizer = tokenizer
        self.model_name = model_name 
        self.yaml_data = yaml.safe_load(open(template_yaml_path, "r", encoding="utf-8"))
        self._precompute_static_components()

    def _precompute_static_components(self):
        """
        설명: System Prompt, Descriptions per Task, Output Format등을 미리 계산하여 전체 maxlen 길이 계산에 사용합니다
        """
        self.static_system_prompt = self.yaml_data.get(f"system_prompt_{self.task.value}", "")
        desc = ""
        if self.mode_config.enable_description:
            summary = self.mode_config.summarize_description
            d_key = "detailed" if not summary else "summarized"

            # Task 타입에 맞게 설명과 Relation List를 YAML에서 가져와 결합합니다.
            if self.task == DataGenerationTask.NER_NER:
                desc = self.yaml_data.get(f"ner_des_{d_key}") + "\n\n" + self.yaml_data.get("ner_ner_relation_list")
            else:  # NER_BEE / NER_BEE_TRUE_ONLY
                desc = self.yaml_data.get(f"ner_des_{d_key}") + "\n\n" + self.yaml_data.get(f"bee_des_{d_key}")
        self.static_description = desc
        
        # Output format selection — branches on (task, reasoning).
        # YAML provides four variants:
        #   output_format_w_keys                (NER-NER, non-reasoning)
        #   output_format_reasoning             (NER-NER, reasoning on)
        #   output_format_w_keys_ner_bee        (NER-BEE, non-reasoning)
        #   output_format_w_keys_ner_bee_reasoning (NER-BEE, reasoning on)
        is_bee = self.task in [DataGenerationTask.NER_BEE, DataGenerationTask.NER_BEE_TRUE_ONLY]
        use_reasoning = bool(getattr(self.mode_config, "reasoning", False))
        if is_bee:
            fmt_key = "output_format_w_keys_ner_bee_reasoning" if use_reasoning else "output_format_w_keys_ner_bee"
        else:
            fmt_key = "output_format_reasoning" if use_reasoning else "output_format_w_keys"
            
        self.static_output_format = self.yaml_data.get(fmt_key, "").strip()

    def compile_prompts(self, entry: dict, fewshot_samples: str = "") -> dict:
        """
        설명: 정적 컴포넌트들과 동적 Input/Output 데이터를 결합하여 프롬프트의 기본 뼈대 구성
        """
        user_sections = []
        if self.static_description: user_sections.append(f"### Description:\n{self.static_description}")
        if fewshot_samples: user_sections.append(f"### Few-shot Examples:\n{fewshot_samples}")
        if self.static_output_format: user_sections.append(f"### Output Format:\n{self.static_output_format}")
        if entry.get("input"): user_sections.append(f"### Input:\n{entry.get('input')}")
        
        out_val = entry.get('output', [])
        if not isinstance(out_val, str):
            out_val = json.dumps(out_val, indent=2, ensure_ascii=False)
            
        ans = f"### Response:\n{out_val}"
        return {"sys": self.static_system_prompt, "user": "\n\n".join(user_sections), "ans": ans}
        
    def _apply_chat_template(self, prompt_data: dict) -> dict:
        """
        설명: 토크나이저의 apply_chat_template에 위임. Gemma4 / 표준 모델 모두 단일 경로.
        Gemma4: enable_thinking 인자로 think 토큰 자동 처리 (chat_template.jinja의 strip_thinking 매크로 활용)
        BOS 토큰은 chat_template이 자동 prepend (수동 strip 금지)
        """
        if not self.tokenizer:
            return {"text": f"{prompt_data['sys']}\n{prompt_data['user']}\n---\n{prompt_data['ans']}"}

        sys_content = f"### System Prompt:\n{prompt_data['sys']}" if prompt_data['sys'] else ""
        user_content = prompt_data['user'] or ""
        ans_content = prompt_data['ans'] or ""

        messages = []
        if sys_content:
            messages.append({"role": "system", "content": sys_content})
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": ans_content})

        kwargs = {"tokenize": False, "add_generation_prompt": False}
        # Gemma4 chat template은 enable_thinking 인자로 think 토큰을 system 턴에 자동 삽입/제거
        is_gemma4 = "gemma4" in (self.model_name or "").lower() or "gemma-4" in (self.model_name or "").lower()
        if is_gemma4:
            kwargs["enable_thinking"] = bool(self.mode_config.reasoning)

        out = self.tokenizer.apply_chat_template(messages, **kwargs)

        # 비-reasoning에서 잔여 legacy <think> 태그 제거 (안전망)
        if not self.mode_config.reasoning:
            out = re.sub(r'<think>.*?</think>\n?', '', out, flags=re.DOTALL)

        return {"text": out}

#################################################################################################
# HELPERS
#################################################################################################
class TokenCounter:
    def __init__(self, model_path: str):
        from transformers import AutoTokenizer
        self.tk = AutoTokenizer.from_pretrained(model_path)
    def count(self, data: Any) -> int:
        return len(self.tk(json.dumps(data, ensure_ascii=False) if not isinstance(data, str) else data, add_special_tokens=False)["input_ids"])

def calculate_template_overhead(prompt_compiler, tokenizer, fewshot_text: str = "") -> int:
    """
    Returns token count of the empty-input template + (optional) few-shot block.

    Args:
        fewshot_text: If the mode uses few-shot, pass a realistic fs_text so the
                      overhead reflects what actually appears in the prompt.
                      Prior version used "" here, underestimating overhead by the
                      full few-shot block (~2000-3000 tokens for full_* modes).
    """
    dummy = {"id": "overhead", "input": "", "output": []}
    # Only pass few-shot if the mode is configured to use it; otherwise the
    # template drops the "### Few-shot Examples:" section entirely.
    mode_uses_fewshot = getattr(prompt_compiler.mode_config, "few_shot", False)
    fs = fewshot_text if mode_uses_fewshot else ""
    compiled = prompt_compiler.compile_prompts(dummy, fs)
    templated = prompt_compiler._apply_chat_template(compiled)
    return len(tokenizer.encode(templated["text"])) + 20

def run_adaptive_chunking(entry, chunker, tc, task_enum, safe_token_limit, thresholds):
    if tc.count(entry) <= safe_token_limit: return [entry]
    for t in thresholds:
        chunks = chunker.chunk_by_relations(entry, target_task=task_enum, entity_threshold=t)
        if chunks and all(tc.count(c) <= safe_token_limit for c in chunks): return chunks
    return chunker.chunk_by_relations(entry, target_task=task_enum, entity_threshold=thresholds[-1])


def _tag_name_from_tagged_str(tagged: str) -> str:
    """Extract the tag name (e.g., 'PER' from '[PER1]Alice[/PER1]') without numeric suffix."""
    if not tagged: return ""
    m = re.search(r'\[([A-Z_][A-Z_0-9/]*?)(\d+)\]', tagged)
    return m.group(1) if m else ""


def build_sampling_profile(chunk_entry, task_enum):
    """
    Build a compact per-chunk profile used for diversity-aware quota sampling.
    Stored as a JSON string in the chunk row so it survives Dataset.map with num_proc.

    Fields:
      - rel_labels: set of distinct relation strings in this chunk (binary for NER-BEE)
      - rel_counts: {relation: count} for head-cap calculation
      - group_pairs: set of "SUBJ_GROUP>OBJ_GROUP" strings (entity-type pair coverage)
      - subj_groups, obj_groups: counters per side (overall entity balance)
      - chunk_pair_count: number of candidate pairs in this chunk
    """
    cands = chunk_entry.get("candidate_pairs", [])
    metas = chunk_entry.get("meta_info", []) or []
    meta_by_pair = {m.get("pair_id"): m for m in metas}

    rel_labels = set()
    rel_counts = {}
    group_pairs = set()
    subj_groups = {}
    obj_groups = {}

    is_bee = task_enum in (DataGenerationTask.NER_BEE, DataGenerationTask.NER_BEE_TRUE_ONLY)

    for cand in cands:
        raw_rel = cand.get("relation", "NO_RELATION")
        # Normalize just as the output path does (negatives → "NO_RELATION")
        if isinstance(raw_rel, str) and raw_rel.strip().lower().replace(" ", "_") in NEGATIVE_VARIANTS:
            raw_rel = "NO_RELATION"
        if raw_rel is None:
            raw_rel = "NO_RELATION"

        if is_bee:
            rel_key = "__bee_true__" if raw_rel != "NO_RELATION" else "__bee_false__"
        else:
            rel_key = raw_rel

        rel_labels.add(rel_key)
        rel_counts[rel_key] = rel_counts.get(rel_key, 0) + 1

        meta = meta_by_pair.get(cand.get("pair_id"), {}) or {}
        s_group = meta.get("subject", {}).get("entity_group") or _tag_name_from_tagged_str(str(cand.get("subject", "")))
        o_group = meta.get("object", {}).get("entity_group") or _tag_name_from_tagged_str(str(cand.get("object", "")))
        s_group = s_group or "UNK"
        o_group = o_group or "UNK"

        group_pairs.add(f"{s_group}>{o_group}")
        subj_groups[s_group] = subj_groups.get(s_group, 0) + 1
        obj_groups[o_group] = obj_groups.get(o_group, 0) + 1

    return json.dumps({
        "rel_labels": sorted(rel_labels),
        "rel_counts": rel_counts,
        "group_pairs": sorted(group_pairs),
        "subj_groups": subj_groups,
        "obj_groups": obj_groups,
        "chunk_pair_count": len(cands),
    }, ensure_ascii=False)

import json
import random

class GenerateFewShotSamples:
    def __init__(self, data_dict, seed=42, allowed_origin_ids=None):
        """
        allowed_origin_ids: set of origin_ids eligible for few-shot pool.
            If provided, chunks whose origin_id is NOT in this set are filtered out
            (used to restrict pool to train split only, preventing val/test leakage).
        """
        if allowed_origin_ids is not None:
            allowed = set(allowed_origin_ids)
            self.data_dict = {
                task: [c for c in chunks if c.get("origin_id") in allowed]
                for task, chunks in data_dict.items()
            }
        else:
            self.data_dict = data_dict
        self.random = random.Random(seed)
        self.RARE_RELATIONS = {
            "instance_of", "addresses", "applied_by", "purchases", "provided_to",
            "gifted_by", "frequency_of_use", "purchased_by", "benefits_user",
            "gifted_to", "sells", "uses", "has_part", "addressed_by_treatment",
            "addressed_to", "belongs_to", "described_by", "not_used_by",
            "requires", "perceives", "targeted_at", "available_to", "available_in",
            "causes", "experiences", "caused_by", "provided_by", "has_instance",
            "variant_of", "owns", "treats", "price_of", "information_to",
            "information_from", "sold_by", "required_by", "targeted_by",
            "child_of", "parent_of", "brand_of", "family_member_of"
        }

    def generate_by_pairs(self, task, min_pairs=8, max_pairs=12, exclude_origin_id=None,
                          prioritize_rare=True, filter_relations=None):
        """
        Args:
            prioritize_rare: For NER-NER, prefer chunks containing rare relations.
                For NER-BEE (is_relational binary), this flag selects chunks that
                have at least one 'true' label (positive pair) to ensure few-shot
                shows non-trivial examples, not all-NO_RELATION.
            exclude_origin_id: chunks sharing this origin_id are excluded
                from the pool (prevents self-demonstration leakage for the current row).
            filter_relations: optional set of relation strings. If provided, only
                chunks whose output contains at least one of these relations are
                eligible. Used by the gold-first hybrid wrapper to pull rare-only
                supplement examples from the main pool.
        """
        chunks = self.data_dict.get(task, [])
        if not chunks: return []

        if exclude_origin_id is not None:
            chunks = [c for c in chunks if c.get("origin_id") != exclude_origin_id]
            if not chunks: return []

        if filter_relations:
            needles = set(filter_relations)
            chunks = [
                c for c in chunks
                if any(f'"{r}"' in c.get("output", "") for r in needles)
            ]
            if not chunks: return []

        # Task-aware "priority" bucket:
        #   NER-NER: chunks containing rare relation string
        #   NER-BEE / NER-BEE_TRUE_ONLY: chunks containing "is_relational": "true"
        #     (positive example presence — otherwise few-shot is all-NO_RELATION)
        is_bee_task = task in ("ner_bee", "ner_bee_true_only")

        rare_chunks = []
        common_chunks = []
        for c in chunks:
            out_str = c.get('output', "")
            if is_bee_task:
                has_priority = '"is_relational": "true"' in out_str
            else:
                has_priority = any(f'"{r}"' in out_str for r in self.RARE_RELATIONS)
            if has_priority:
                rare_chunks.append(c)
            else:
                common_chunks.append(c)

        self.random.shuffle(rare_chunks)
        self.random.shuffle(common_chunks)

        # When prioritize_rare is False, just interleave/shuffle everything.
        if not prioritize_rare:
            interleaved = list(chunks)
            self.random.shuffle(interleaved)
            rare_chunks = interleaved
            common_chunks = []

        selected_samples = []
        current_pair_count = 0

        # 🎯 Count pairs dynamically based on the output text
        for chunk in rare_chunks:
            chunk_pairs = chunk.get('output', "").count('"subject":')
            if chunk_pairs == 0: continue # Skip empties

            if current_pair_count + chunk_pairs <= max_pairs:
                selected_samples.append(chunk)
                current_pair_count += chunk_pairs

            if current_pair_count >= min_pairs:
                break

        # Fallback to common chunks if rare chunks didn't fill the 8-12 gap
        if current_pair_count < min_pairs:
            for chunk in common_chunks:
                chunk_pairs = chunk.get('output', "").count('"subject":')
                if chunk_pairs == 0: continue

                if current_pair_count + chunk_pairs <= max_pairs:
                    selected_samples.append(chunk)
                    current_pair_count += chunk_pairs
                if current_pair_count >= min_pairs:
                    break

        return selected_samples

    def generate_gold_first(self, task, min_pairs=8, max_pairs=12, exclude_origin_id=None,
                            prioritize_rare=True, supplementary_generator=None):
        """
        Gold-first few-shot selection: draw primarily from `self` (gold pool),
        then top up with rare-class chunks from `supplementary_generator` (main pool)
        only for rare relations that gold couldn't cover.

        Behavior:
          1. Pick up to `max_pairs` worth of samples from gold (this generator).
          2. If NER-NER and fewer than `max_pairs` pairs accumulated, identify
             which rare RELATIONS are missing and request a top-up from the
             supplementary pool restricted to those missing relations.
          3. Supplementary picks honor the same `exclude_origin_id` (though for
             gold mode the main pool's origin space is disjoint anyway).

        If `supplementary_generator` is None this degrades to the plain
        `generate_by_pairs` call — useful for gold-only eval scenarios.
        """
        primary = self.generate_by_pairs(
            task,
            min_pairs=min_pairs,
            max_pairs=max_pairs,
            exclude_origin_id=exclude_origin_id,
            prioritize_rare=prioritize_rare,
        )

        if supplementary_generator is None:
            return primary

        # Count pairs currently covered by primary, figure headroom
        def _pair_count(samples):
            return sum(s.get("output", "").count('"subject":') for s in samples)

        covered_pairs = _pair_count(primary)
        headroom = max_pairs - covered_pairs
        if headroom <= 0:
            return primary

        # Which rare relations are still missing from primary?
        covered_relations = set()
        for s in primary:
            out_str = s.get("output", "")
            for r in self.RARE_RELATIONS:
                if f'"{r}"' in out_str:
                    covered_relations.add(r)
        missing_rare = self.RARE_RELATIONS - covered_relations

        if not missing_rare:
            return primary

        # Pull rare-only supplement from the main pool
        supplement = supplementary_generator.generate_by_pairs(
            task,
            min_pairs=0,
            max_pairs=headroom,
            exclude_origin_id=exclude_origin_id,
            prioritize_rare=True,
            filter_relations=missing_rare,
        )

        if supplement:
            primary = list(primary) + list(supplement)
        return primary

    def format(self, samples):
        """
        🎯 Restored & Improved: Merges multiple chunks into a clean Example block.
        """
        if not samples:
            return ""
            
        lines = []
        for i, s in enumerate(samples, 1):
            # Use original 'input' string from the preprocessor
            input_val = s.get("input", "")
            
            # Ensure output is a pretty JSON string
            out_val = s.get("output", "[]")
            if not isinstance(out_val, str):
                out_val = json.dumps(out_val, ensure_ascii=False, indent=2)
                
            lines.append(f"### Example {i}")
            lines.append(input_val)
            lines.append(f"### Example {i} Response")
            lines.append(out_val)
            lines.append("-" * 30) # Separator for multi-chunk few-shots
            
        return "\n".join(lines)
